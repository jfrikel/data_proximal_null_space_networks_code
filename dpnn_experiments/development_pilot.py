"""Sequential development study; stage processes isolate CUDA memory and archives."""
import argparse
import copy
import json
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def resolved(plan):
    """Resolve all scientific choices before any dataset or optimizer is run."""
    if plan['schema_version'] != 1:
        raise ValueError('Unknown development plan version')
    if set(plan['counts']) != {'train', 'validation', 'test'}:
        raise ValueError('Three explicit disjoint split counts required')
    if any(type(n) is not int or n < 1 for n in plan['counts'].values()):
        raise ValueError('Nonempty split counts required; test is reserved, never evaluated')
    if not plan['methods'] or len(set(plan['methods'])) != len(plan['methods']) or not set(plan['methods']) <= {'RES', 'NSN', 'DP', 'DP_TWO'}:
        raise ValueError('Invalid methods')
    if set(plan['starts']) != {'tikhonov', 'tv'}:
        raise ValueError('Matched Tikhonov and TV starts required')
    base = copy.deepcopy(plan['training_configuration']) if 'training_configuration' in plan else read(SOURCE / plan['training_template'])
    data = copy.deepcopy(plan['data_configuration']) if 'data_configuration' in plan else read(SOURCE / plan['data_template'])
    base['geometry'] = copy.deepcopy(data['geometry'])
    base['study_id'] = data['study_id'] = plan['study_id']
    data['purpose'] = 'Independent development population; reserved internal test never evaluated'
    data['dataset']['split_ids'] = {
        split: [f"{plan['study_id']}/{split}/{i:06d}" for i in range(count)]
        for split, count in plan['counts'].items()}
    data['randomness'].update(plan['data_seeds'])
    data['computation'].update(device=plan['device'], cpu_threads=plan['cpu_threads'],
                               relative_noise_norm=plan['relative_noise_norm'])
    data['computation']['tikhonov'] = {k: v for k, v in plan['starts']['tikhonov'].items() if k != 'kind'}
    base['computation'].update(device=plan['device'], cpu_threads=plan['cpu_threads'])
    base['training'].update(epochs=plan['epochs'], batch_size=plan['batch_size'])
    base['selection'].update(patience=plan['patience'], min_delta=plan['min_delta'])
    base['scheduler'].update(plan['scheduler'])
    base['optimizer']['lr'] = plan['learning_rate']
    base['logging'].update(weights_every_step=False, checkpoint_every_steps=plan['checkpoint_every_steps'])
    base['randomness'].update(plan['training_seeds'])
    base['execution'].update(resume=None, stop_after_step=None, injected_io_failure_step=None)
    if 'projection' in plan:
        base['projection'] = copy.deepcopy(plan['projection'])
    data['operator_sha256'] = plan['operator_sha256']
    if base['projection']['kind'] == 'block' and base['projection']['operator_sha256'] != plan['operator_sha256']:
        raise ValueError('Operator/projection hash mismatch')
    return data, base


def start_names(plan):
    names = plan.get('selected_starts', list(plan['starts']))
    if not names or len(set(names)) != len(names) or not set(names) <= set(plan['starts']):
        raise ValueError('Invalid selected starts')
    if 'external_inputs' not in plan and set(names) != set(plan['starts']):
        raise ValueError('Partial workflow requires verified external inputs')
    return names


def external_inputs(plan):
    """Fail closed before training; external caches never invoke a solver fallback."""
    from dpnn_experiments.training_archive import verify_reference, file_reference
    from dpnn_experiments.cached_starts import EvaluationStartCache
    from dpnn_experiments.provenance import digest
    inputs = plan['external_inputs']
    if 'continuation_of' in plan:
        verify_reference(plan['continuation_of']['plan'])
        verify_reference(plan['continuation_of']['launch_manifest'])
    if set(inputs['start_caches']) != set(start_names(plan)):
        raise ValueError('External cache set must match selected starts')
    data, _ = resolved(plan)
    def archive(reference):
        path = verify_reference(reference)
        root = path.parent
        completion = read(root/'completion.json')
        if (completion['status'] != 'completed'
                or completion['manifest_sha256'] != digest(root/'manifest.json')
                or completion['artifacts_sha256'] != digest(root/'artifacts.json')):
            raise ValueError('External archive incomplete or altered')
        entries = read(root/'artifacts.json')['files']
        if not any(e['path'] == path.name and e['sha256'] == reference['sha256'] for e in entries):
            raise ValueError('External dataset is not in its archive')
        for e in entries:
            item = (root/e['path']).resolve()
            if root.resolve() not in item.parents:
                raise ValueError('Invalid archive member path')
            verify_reference({'path': str(item), 'sha256': e['sha256'], 'bytes': e['bytes']})
        return path, read(path)
    raw, binding = archive(inputs['raw_dataset'])
    if (binding['split_ids'] != data['dataset']['split_ids']
            or binding['geometry'] != data['geometry']
            or binding['operator_sha256'] != plan['operator_sha256']):
        raise ValueError('External raw dataset differs from resolved plan')
    original_data = binding['configuration']
    if (original_data['randomness'] != data['randomness']
            or original_data['study_id'] != data['study_id']
            or original_data['computation']['relative_noise_norm'] != plan['relative_noise_norm']):
        raise ValueError('External population seeds or noise differ from plan')
    records = {r['id']: r for r in binding['records']}
    flat = [i for ids in binding['split_ids'].values() for i in ids]
    if len(flat) != len(set(flat)) or len(records) != len(binding['records']) or set(flat) != set(records):
        raise ValueError('Invalid external dataset IDs')
    caches = {}
    for name in start_names(plan):
        cache_path, cached = archive(inputs['start_caches'][name])
        if cached.get('parent_dataset') != inputs['raw_dataset']:
            raise ValueError('External cache parent mismatch')
        # Validate every training/validation sample against its immutable raw data.
        # Reserved test samples are hash-checked only, never evaluated.
        for split in ('train', 'validation'):
            ids = binding['split_ids'][split]
            cache = EvaluationStartCache({**plan['starts'][name], 'cache': inputs['start_caches'][name]},
                                         binding, data['geometry'], ids, split)
            for ident in ids:
                if records[ident]['split'] != split:
                    raise ValueError('Raw record split mismatch')
                cache.load(ident, records[ident], 'cpu')
        caches[name] = cache_path.parent
    return raw.parent, caches, {'raw_dataset': file_reference(raw),
        'start_caches': inputs['start_caches'], 'validated_counts': plan['counts'],
        'test_quality_evaluated': False, 'solver_recomputation': False}


def preflight(plan):
    import torch
    import numpy as np
    from dpnn_experiments.training import validate_config
    from dpnn_experiments.gpu_benchmark import reference_metadata
    from dpnn_experiments.training_archive import verify_reference
    from dpnn_experiments.network_layers import checked_archive_file
    data, base = resolved(plan)
    names = start_names(plan)
    external = external_inputs(plan)[2] if 'external_inputs' in plan else None
    validate_config(base)
    ref = reference_metadata(Path(plan['operator_root']))
    if ref['geometry'] != data['geometry'] or ref['files']['operator.npz'] != plan['operator_sha256']:
        raise ValueError('Missing or incompatible operator archive')
    p = base['projection']
    if p['kind'] == 'block':
        # This hashes the full Q file without loading its ~20 GB into memory.
        for root in (p['factor_run'], p['verification_run']):
            for name in ('manifest.json', 'completion.json'):
                checked_archive_file(Path(root), name)
            if read(Path(root)/'completion.json')['status'] != 'completed':
                raise ValueError('Factor/verification archive not completed')
        checked_archive_file(Path(p['factor_run']), 'row_space_basis.npy', p['basis_sha256'])
        checked_archive_file(Path(p['verification_run']), 'summary.json', p['verification_summary_sha256'])
        # Detailed basis/operator contract is also enforced when building each model.
    else:
        verify_reference(p['reference'])
    if plan['device'] == 'cuda:0':
        if not torch.cuda.is_available():
            raise RuntimeError('Selected Python environment cannot use CUDA')
        device_name = torch.cuda.get_device_name(0)
    elif plan['device'] == 'cpu':
        device_name = 'CPU technical fixture'
    else:
        raise ValueError('Use cuda:0 with CUDA_VISIBLE_DEVICES selecting the physical GPU')
    return {'passed': True, 'torch': str(torch.__version__), 'numpy': np.__version__,
            'cuda': torch.version.cuda, 'device': device_name, 'counts': plan['counts'],
            'training_jobs': len(plan['methods'])*len(names), 'external_inputs': external,
            'scientific_test_evaluation': False}


def stage(kind, config, root, operator_root=None):
    if kind == 'data':
        from dpnn_experiments.pilot_data import prepare
        return prepare(config, root, Path(operator_root))
    if kind == 'start':
        from dpnn_experiments.start_preparation import execute
    elif kind == 'train':
        from dpnn_experiments.training import execute
    elif kind == 'evaluate':
        from dpnn_experiments.evaluator import execute
    else:
        raise ValueError(kind)
    return execute(config, root)


def workflow(plan, root, run_stage):
    """run_stage retains each attempt, skips verified successes and resumes training."""
    from dpnn_experiments.training_archive import file_reference, verify_reference
    from dpnn_experiments.evaluator import METRICS
    from dpnn_experiments.provenance import write_json
    root = Path(root)
    data_config, base = resolved(plan)
    names = start_names(plan)
    if 'external_inputs' in plan:
        raw, caches, verified = external_inputs(plan)
        write_json(root/'external_inputs_verified.json', verified)
    else:
        raw = run_stage('data', 'data', data_config, plan['operator_root'])
        caches = {}
    binding = read(raw/'dataset.json')
    evaluation_index = []
    for start_name in names:
        settings = plan['starts'][start_name]
        sc = {'schema_version': 1, 'purpose': 'Fixed development starts; no parameter selection',
              'geometry': data_config['geometry'], 'randomness': data_config['randomness'],
              'computation': base['computation'], 'operator_root': plan['operator_root'],
              'dataset': {'reference': file_reference(raw/'dataset.json')}, 'start': settings}
        cache = caches[start_name] if 'external_inputs' in plan else run_stage('start_'+start_name, 'start', sc, None)
        for method in plan['methods']:
            tc = copy.deepcopy(base)
            tc['purpose'] = f"Development training {method}/{start_name}; no scientific test access"
            tc['model']['method'] = method
            tc['dataset']['reference'] = file_reference(cache/'dataset.json')
            tc['dataset']['used_ids'] = {k: binding['split_ids'][k] for k in ('train', 'validation')}
            trained = run_stage('train_'+start_name+'_'+method, 'train', tc, None)
            for choice in ('best_mse', 'best_ssim'):
                best = read(trained/(choice+'.json'))
                # A resumed segment can point to an earlier segment's best weight.
                # The evaluator must receive the actual owning manifest.
                owner = verify_reference(best['weights']).parent.parent
                ec = {'schema_version': 1, 'purpose': f'Development validation {method}/{start_name}/{choice}',
                      'dataset': {'reference': file_reference(raw/'dataset.json')},
                      'geometry': tc['geometry'], 'randomness': tc['randomness'], 'computation': tc['computation'],
                      'model_config': tc, 'models': [{'method': method, 'trained_start': start_name,
                        'training_manifest': file_reference(owner/'manifest.json'), 'weights': best['weights']}],
                      'starts': [{**settings, 'cache': file_reference(cache/'dataset.json')}],
                      'beta_multipliers': plan['evaluation_beta_multipliers'], 'split': 'validation',
                      'ids': binding['split_ids']['validation'],
                      'diagnostic_ids': binding['split_ids']['validation'][:plan['diagnostic_objects']],
                      'allow_start_transfer': False, 'metrics': {**METRICS, **tc['metrics']},
                      'tolerances': plan['evaluation_tolerances'], 'frozen_protocol': None}
                evaluated = run_stage('eval_'+start_name+'_'+method+'_'+choice, 'evaluate', ec, None)
                evaluation_index.append({'method': method, 'start': start_name, 'selection': choice,
                    'training_run': str(trained), 'weight_owner': str(owner), 'evaluation_run': str(evaluated),
                    'summary': file_reference(evaluated/'summary.json')})
                write_json(root/'evaluation_index.json', evaluation_index)
    return {'completed': True, 'training_jobs': len(plan['methods'])*len(names),
            'evaluation_jobs': len(evaluation_index), 'scientific_test_evaluated': False,
            'interpretation': 'One-seed development validation only, not a final method ranking',
            'evaluations': evaluation_index}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', required=True, type=Path)
    args = p.parse_args()
    spec = read(args.stage)
    from dpnn_experiments.provenance import write_json
    if spec['kind'] == 'preflight':
        write_json(spec['result'], preflight(spec['config']))
        return
    out = stage(spec['kind'], spec['config'], Path(spec['root']), spec.get('operator_root'))
    completion = read(out/'completion.json')
    from dpnn_experiments.training_archive import file_reference
    write_json(spec['result'], {'run': str(out), 'status': completion['status'],
        'completion': file_reference(out/'completion.json'), 'artifacts': file_reference(out/'artifacts.json')})
    if completion['status'] != 'completed':
        raise SystemExit(75)


if __name__ == '__main__':
    main()
