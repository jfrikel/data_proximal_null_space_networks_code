"""Fail-closed bindings for main-study approvals and validation selections."""
import copy
import json
import math
import socket
from datetime import datetime, timezone, timedelta
from pathlib import Path

from dpnn_experiments.development_pilot import resolved
from dpnn_experiments.provenance import digest
from dpnn_experiments.training_archive import file_reference, verify_reference

SOURCE = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def implementation():
    """Path-independent fingerprint also matches a launcher's frozen source copy."""
    return {str(p.relative_to(SOURCE)): digest(p)
            for folder in ('dpnn_experiments', 'scripts', 'src', 'tests', 'tools')
            for p in sorted((SOURCE/folder).rglob('*.py')) if '__pycache__' not in p.parts}


class SealedArchive:
    def __init__(self, root, statuses=('completed',)):
        self.root = Path(root).resolve()
        completion = read(self.root/'completion.json')
        if completion.get('status') not in statuses:
            raise ValueError('Archive is not closed in an allowed state: '+str(root))
        for name, key in (('manifest.json', 'manifest_sha256'), ('artifacts.json', 'artifacts_sha256')):
            if digest(self.root/name) != completion.get(key):
                raise ValueError('Archive seal changed: '+str(root))
        entries = read(self.root/'artifacts.json')['files']
        self.entries = {e['path']: e for e in entries}
        if len(entries) != len(self.entries):
            raise ValueError('Duplicate archive members')
        for name in self.entries:
            if self.root not in (self.root/name).resolve().parents:
                raise ValueError('Archive member escapes root')
        self.configuration = read(self.member('manifest.json'))['configuration']

    def member(self, name):
        e = self.entries[name]
        return verify_reference({'path': str(self.root/name), 'bytes': e['bytes'], 'sha256': e['sha256']})

    def reference(self, ref):
        path = verify_reference(ref).resolve()
        self.member(str(path.relative_to(self.root)))
        return path


def production_acceptance(root, a):
    profile_ref = file_reference(root/'profile.json')
    profile = read(verify_reference(profile_ref))
    if a.get('profile') != profile_ref:
        raise ValueError('Acceptance is not bound to this profile')
    for field in ('quota_checked', 'backup_checked', 'gpu_allocation_confirmed'):
        if a.get(field) is not True:
            raise ValueError('Acceptance missing: '+field)
    reports = {key: read(verify_reference(a[key])) for key in (
        'pilot_review', 'budget_report', 'qualification_report', 'parallel_rehearsal_report',
        'scientific_decision', 'quota_report', 'backup_report', 'gpu_allocation_report')}
    if reports['pilot_review'].get('passed') is not True:
        raise ValueError('Pilot review failed')
    scientific = reports['scientific_decision']
    if scientific.get('pilot_accepted') is not True or scientific.get('main_profile') != profile_ref:
        raise ValueError('Scientific decision is missing or for a different profile')
    budget = reports['budget_report']
    hours = budget['estimated_total_hours_with_reserve']
    if (budget.get('profile') != profile_ref or not math.isfinite(hours) or hours <= 0
            or a['maximum_wall_hours'] != profile['maximum_wall_hours']
            or hours > a['maximum_wall_hours']):
        raise ValueError('Budget/profile/wall-time mismatch')
    for key in ('qualification_report', 'parallel_rehearsal_report'):
        report = reports[key]
        if report.get('passed') is not True or report.get('implementation') != implementation():
            raise ValueError('Missing qualification of current implementation: '+key)
    if reports['parallel_rehearsal_report'].get('gpu_count') != 4:
        raise ValueError('Four-GPU rehearsal evidence missing')
    for key in ('quota_report', 'backup_report', 'gpu_allocation_report'):
        if reports[key].get('profile') != profile_ref:
            raise ValueError('Resource evidence/profile mismatch: '+key)
    quota = reports['quota_report']
    if (quota.get('passed') is not True or quota.get('budget_report') != a['budget_report']
            or quota.get('free_gib', 0) < budget['estimated_storage_gib_with_reserve']+20):
        raise ValueError('Quota/capacity evidence failed')
    backup = reports['backup_report']
    if any(backup.get(k) is not True for k in ('passed', 'independent_full_backup_verified', 'restore_verified')):
        raise ValueError('Independent backup and restore verification required')
    verify_reference(backup['verification_evidence'])
    allocation = reports['gpu_allocation_report']
    if (allocation.get('allocated') is not True or not allocation.get('authority')
            or not allocation.get('allocation_id') or allocation.get('hostname') != socket.gethostname()):
        raise ValueError('Authoritative GPU allocation for this host required')
    verify_reference(allocation['authorization_evidence'])
    ids = allocation.get('gpu_uuids', [])
    if len(ids) != 4 or len(set(ids)) != 4 or any(not x.startswith('GPU-') for x in ids):
        raise ValueError('Allocation must name four distinct GPU UUIDs')
    until = datetime.fromisoformat(allocation['valid_until_utc'])
    if until.tzinfo is None or until < datetime.now(timezone.utc)+timedelta(hours=hours):
        raise ValueError('GPU allocation does not cover the estimated execution window')
    return reports


def check_allocation(a, ids, allocation):
    report = read(verify_reference(a['gpu_allocation_report']))
    if report['gpu_uuids'] != ids or report['allocation_id'] != allocation:
        raise ValueError('Requested GPUs/allocation differ from the accepted reservation')


def preparation_references(root):
    return {name: file_reference(root/(name+'.json')) for name in
            ('profile', 'base_plan', 'protocol_design', 'split_ids', 'inputs', 'workers')}


def verified_selections(root, launcher):
    """Reconstruct expected validation contracts from the frozen worker plans."""
    from dpnn_experiments.evaluator import METRICS, protocol, validate
    controller = root/'controller'
    manifest = launcher.verify_snapshot(controller)
    control = read(verify_reference(manifest['controller_control']))
    if control['preparation_references'] != preparation_references(root):
        raise ValueError('Main-study preparation changed since launch')
    for ref in control['plans']:
        verify_reference(ref)
    profile = read(root/'profile.json'); design = read(root/'protocol_design.json')
    splits = read(root/'split_ids.json'); policy = profile['test_policy']
    if (design['profile'] != file_reference(root/'profile.json')
            or design['split_ids'] != file_reference(root/'split_ids.json') or design['design'] != policy):
        raise ValueError('Frozen design/profile mismatch')
    if policy['checkpoint_selections'] != ['best_mse', 'best_ssim']:
        raise ValueError('Expected both pre-specified checkpoint selections')
    betas = [policy['primary_beta_multiplier'], *policy['diagnostic_beta_multipliers']]
    workers = read(root/'workers.json')
    expected_names = {f'seed{i}_{s}' for i in range(2) for s in ('tikhonov', 'tv')}
    if len(workers) != 4 or {w['name'] for w in workers} != expected_names:
        raise ValueError('Expected four distinct seed/start workers')
    inputs = read(root/'inputs.json'); rows = []; evidence = []
    for item in workers:
        study = root/'workers'/item['name']; plan = read(verify_reference(item['plan']))
        seed_index = int(item['name'][4]); start = item['name'][6:]
        expected_plan = read(root/'base_plan.json')
        # bind-inputs may increase this operational capacity guard from its budget.
        expected_plan['minimum_free_gib'] = plan['minimum_free_gib']
        expected_plan.update(training_seeds=profile['training_seed_sets'][seed_index], selected_starts=[start],
            external_inputs={'raw_dataset': inputs['raw_dataset'], 'start_caches': {start: inputs['start_caches'][start]}})
        if plan != expected_plan:
            raise ValueError('Worker plan differs from prepared design: '+item['name'])
        launch = launcher.verify_snapshot(study)
        if launch['source_files'] != manifest['source_files']:
            raise ValueError('Worker source differs from frozen controller source')
        if read(verify_reference(launch['plan'])) != plan or read(study/'status.json')['status'] != 'completed':
            raise ValueError('Worker is not completed with the expected plan')
        data, base = resolved(plan)
        if (data['dataset']['split_ids'] != splits or plan['counts'] != profile['counts']
                or plan['epochs'] != profile['epochs'] or plan['data_seeds'] != profile['data_seeds']
                or set(plan['methods']) != {'RES', 'NSN', 'DP', 'DP_TWO'}
                or len(plan['methods']) != 4 or set(plan['evaluation_beta_multipliers']) != set(betas)):
            raise ValueError('Worker scientific settings differ from profile')
        records = read(study/'evaluation_index.json')
        expected_pairs = {(m, s) for m in plan['methods'] for s in policy['checkpoint_selections']}
        if len(records) != 8 or {(r['method'], r['selection']) for r in records} != expected_pairs:
            raise ValueError('Missing or duplicate validation selections')
        for record in records:
            method, choice = record['method'], record['selection']
            if record['start'] != start:
                raise ValueError('Validation start mismatch')
            train = SealedArchive(record['training_run']); evaluation = SealedArchive(record['evaluation_run'])
            # Require the owning worker/stage, not an unrelated valid archive.
            for archive, label in ((train, f'train_{start}_{method}'), (evaluation, f'eval_{start}_{method}_{choice}')):
                if study.resolve()/'stages'/label not in archive.root.parents:
                    raise ValueError('Archive is outside its designated worker stage')
                sources = read(archive.member('manifest.json'))['source_files']
                actual_sources = {x['path']: x['sha256'] for x in sources if x['path'].endswith('.py')}
                expected_sources = {x['path']: x['sha256'] for x in launch['source_files'] if x['path'].endswith('.py')}
                if actual_sources != expected_sources:
                    raise ValueError('Archive source differs from frozen worker source')
            best = read(train.member(choice+'.json'))
            weights = train.reference(best['weights']); owner = weights.parent.parent
            if Path(record['weight_owner']).resolve() != owner:
                raise ValueError('Selected weight owner mismatch')
            tc = copy.deepcopy(base)
            tc['purpose'] = f'Development training {method}/{start}; no scientific test access'
            tc['model']['method'] = method
            tc['dataset']['reference'] = inputs['start_caches'][start]
            tc['dataset']['used_ids'] = {k: splits[k] for k in ('train', 'validation')}
            actual_training = copy.deepcopy(train.configuration)
            # Resume changes only execution bookkeeping; all scientific keys must agree.
            for key in ('checkpoint_selection', 'dp_contract', 'retention_contract', 'model_evaluation'):
                actual_training.pop(key, None)
            actual_training['execution']['resume'] = None
            if actual_training != tc:
                raise ValueError('Training configuration differs from worker plan')
            if read(train.member('summary.json'))['test_read_count'] != 0:
                raise ValueError('Training accessed reserved test objects')
            ec = {'dataset': {'reference': inputs['raw_dataset']}, 'geometry': tc['geometry'],
                'model_config': tc, 'models': [{'method': method, 'trained_start': start,
                    'training_manifest': file_reference(train.member('manifest.json')), 'weights': best['weights']}],
                'starts': [{**plan['starts'][start], 'cache': inputs['start_caches'][start]}],
                'beta_multipliers': plan['evaluation_beta_multipliers'], 'split': 'validation',
                'ids': splits['validation'], 'diagnostic_ids': splits['validation'][:plan['diagnostic_objects']],
                'allow_start_transfer': False, 'metrics': {**METRICS, **tc['metrics']},
                'tolerances': plan['evaluation_tolerances'], 'computation': tc['computation']}
            config = evaluation.configuration
            validate(config)
            if protocol(config) != protocol(ec) or config['randomness'] != tc['randomness']:
                raise ValueError('Validation protocol differs from selected model/design')
            summary_path = evaluation.reference(record['summary']); summary = read(summary_path)
            if summary_path != evaluation.root/'summary.json' or (summary['split'], summary['test_objects_read'], summary['evaluated_ids']) != ('validation', 0, splits['validation']):
                raise ValueError('Invalid validation summary')
            if read(evaluation.member('effective_protocol.json')) != protocol(ec):
                raise ValueError('Effective validation protocol mismatch')
            config = copy.deepcopy(config)
            for key in ('training', 'model_evaluation'):
                config.pop(key, None)
            config.update(split='test', ids=splits['test'], diagnostic_ids=splits['test'][:policy['diagnostic_objects']],
                frozen_protocol=None, purpose='Independent frozen main-study test; no tuning')
            rows.append((item['name'], method+'_'+choice, config))
            evidence.append({'worker': item['name'], 'method': method, 'selection': choice,
                'training_completion': file_reference(train.root/'completion.json'),
                'evaluation_completion': file_reference(evaluation.root/'completion.json'),
                'selected_weights': best['weights'], 'selection_record': file_reference(train.member(choice+'.json'))})
    return rows, evidence
