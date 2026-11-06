"""Small archived H200 test of TV continuation with actual U-Nets and checkpoint resume."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

from dpnn_experiments.development_pilot import read, resolved, preflight
from dpnn_experiments.training_archive import TrainingArchive, file_reference, verify_reference
from tests.training_qualification import compare_runs
from dpnn_experiments.evaluator import METRICS
from dpnn_experiments.provenance import write_json

SOURCE = Path(__file__).resolve().parents[1]


def run(plan, root):
    root = Path(root).resolve()
    data, base = resolved(plan)
    cfg = {'schema_version': 1, 'purpose': 'External TV cache and DP_TWO checkpoint rehearsal',
           'geometry': data['geometry'], 'dataset': plan['external_inputs'],
           'randomness': base['randomness'], 'computation': base['computation'], 'plan': plan}
    with TrainingArchive(cfg, root/'runs', SOURCE, root/'metadata') as archive:
        write_json(archive.path/'preflight.json', preflight(plan))
        raw_ref = plan['external_inputs']['raw_dataset']
        cache_ref = plan['external_inputs']['start_caches']['tv']
        binding = read(verify_reference(raw_ref))
        tc = copy.deepcopy(base)
        tc['purpose'] = 'Technical DP_TWO TV-cache rehearsal on four training/two validation objects'
        tc['model']['method'] = 'DP_TWO'
        tc['dataset']['reference'] = cache_ref
        tc['dataset']['used_ids'] = {k: binding['split_ids'][k][:n] for k,n in [('train',4),('validation',2)]}
        tc['training'].update(epochs=2, batch_size=2)
        tc['selection'].update(patience=5)
        tc['logging']['checkpoint_every_steps'] = 1
        commands = []
        def stage(label, kind, config, allowed='completed'):
            folder = archive.path/label; folder.mkdir()
            spec = {'kind': kind, 'config': config, 'root': str(folder), 'result': str(folder/'result.json')}
            path = folder/'stage.json'; write_json(path, spec)
            command = [sys.executable, '-u', '-m', 'dpnn_experiments.development_pilot', '--stage', str(path)]
            with (folder/'process.log').open('w') as log:
                result = subprocess.run(command, cwd=SOURCE, stdout=log, stderr=subprocess.STDOUT)
            commands.append({'command': command, 'returncode': result.returncode, 'log': str(folder/'process.log')})
            write_json(archive.path/'commands.json', commands)
            record = read(folder/'result.json')
            assert record['status'] == allowed, record
            assert result.returncode == (75 if allowed == 'interrupted' else 0), result.returncode
            return Path(record['run'])
        uninterrupted = stage('uninterrupted', 'train', tc)
        interrupted_config = copy.deepcopy(tc); interrupted_config['execution']['stop_after_step'] = 1
        interrupted = stage('interrupted', 'train', interrupted_config, 'interrupted')
        resumed_config = copy.deepcopy(tc)
        resumed_config['execution']['resume'] = read(interrupted/'latest_checkpoint.json')
        resumed = stage('resumed', 'train', resumed_config)
        comparison = compare_runs(uninterrupted, interrupted, resumed, 1e-12)
        evaluations = []
        for choice in ('best_mse','best_ssim'):
            best = read(resumed/(choice+'.json'))
            owner = verify_reference(best['weights']).parent.parent
            ec = {'schema_version':1, 'purpose':'Technical TV continuation validation '+choice,
                'dataset':{'reference':raw_ref}, 'geometry':tc['geometry'], 'randomness':tc['randomness'],
                'computation':tc['computation'], 'model_config':tc,
                'models':[{'method':'DP_TWO','trained_start':'tv',
                    'training_manifest':file_reference(owner/'manifest.json'),'weights':best['weights']}],
                'starts':[{**plan['starts']['tv'],'cache':cache_ref}],
                'beta_multipliers':plan['evaluation_beta_multipliers'], 'split':'validation',
                'ids':tc['dataset']['used_ids']['validation'],
                'diagnostic_ids':tc['dataset']['used_ids']['validation'], 'allow_start_transfer':False,
                'metrics':{**METRICS,**tc['metrics']}, 'tolerances':plan['evaluation_tolerances'], 'frozen_protocol':None}
            evaluated = stage(choice, 'evaluate', ec)
            summary = read(evaluated/'summary.json'); assert summary['test_objects_read'] == 0
            evaluations.append(file_reference(evaluated/'summary.json'))
        verify_reference(raw_ref); verify_reference(cache_ref)
        write_json(archive.path/'summary.json', {'passed':True, 'resume_comparison':comparison,
            'training_objects':4,'validation_objects':2,'epochs':2,'method':'DP_TWO',
            'evaluations':evaluations,'scientific_test_evaluated':False,'solver_recomputation':False})
    return archive.path


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--plan',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args(); print(run(read(a.plan), a.root))
