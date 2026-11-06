# SPDX-License-Identifier: GPL-3.0-only
"""Archived external-cache continuation checks, including real checkpoint resume."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import time

from tests.development_qualification import run as qualify_existing
from dpnn_experiments.development_pilot import external_inputs
from dpnn_experiments.training_archive import TrainingArchive, file_reference, verify_reference
from dpnn_experiments.provenance import write_json

SOURCE = Path(__file__).resolve().parents[1]


def run(root):
    root = Path(root).resolve()
    regression = qualify_existing(root/'existing_workflow')
    prior = json.loads((regression/'summary.json').read_text())
    original = Path(prior['study'])
    plan = json.loads((original/'plan.json').read_text())
    raw = next((original/'stages/data').glob('attempt_*/runs/*/dataset.json'))
    cache = next((original/'stages/start_tv').glob('attempt_*/runs/*/dataset.json'))
    plan.update(selected_starts=['tv'], external_inputs={
        'raw_dataset': file_reference(raw), 'start_caches': {'tv': file_reference(cache)}})
    cfg = {'schema_version': 1, 'purpose': 'External cache continuation guards and checkpoint resume',
           'geometry': plan['data_configuration']['geometry'], 'randomness': plan['training_seeds'],
           'computation': {'device': 'cpu', 'dtype': 'float64'},
           'dataset': plan['external_inputs'], 'plan': plan}
    with TrainingArchive(cfg, root/'runs', SOURCE, root/'metadata') as archive:
        before = {str(p): file_reference(p) for p in cache.parent.rglob('*') if p.is_file()}
        _, _, verified = external_inputs(plan)
        write_json(archive.path/'verified_inputs.json', verified)
        rejected = []
        for label in ('hash', 'settings', 'split', 'missing_cache', 'noise', 'seeds'):
            bad = copy.deepcopy(plan)
            if label == 'hash': bad['external_inputs']['start_caches']['tv']['sha256'] = '0'*64
            if label == 'settings': bad['starts']['tv']['max_iterations'] += 1
            if label == 'split': bad['counts']['train'] += 1
            if label == 'missing_cache': bad['external_inputs']['start_caches'] = {}
            if label == 'noise': bad['relative_noise_norm'] *= 2
            if label == 'seeds': bad['data_seeds']['noise_seed'] += 1
            try: external_inputs(bad)
            except ValueError as e: rejected.append({'case': label, 'reason': str(e)})
            else: raise AssertionError('Failed guard: '+label)
        write_json(archive.path/'guards.json', rejected)
        commands = []
        def command(*args):
            cmd = [sys.executable, str(SOURCE/'scripts/development_pilot.py'), *map(str, args)]
            result = subprocess.run(cmd, cwd=SOURCE, capture_output=True, text=True)
            commands.append({'command': cmd, 'returncode': result.returncode,
                             'stdout': result.stdout, 'stderr': result.stderr})
            write_json(archive.path/'commands.json', commands)
            assert result.returncode == 0, result.stderr
        def state(study, terminal):
            until = time.monotonic()+300
            while time.monotonic() < until:
                if (study/'status.json').exists():
                    status = json.loads((study/'status.json').read_text())
                    if status['status'] in terminal: return status
                time.sleep(.05)
            raise TimeoutError(str(study))
        pc = archive.path/'plan.json'; write_json(pc, plan)
        study = archive.path/'continuation'
        command('start', '--config', pc, '--root', study)
        status = state(study, {'completed', 'failed'}); assert status['status'] == 'completed', status
        summary = json.loads((study/'summary.json').read_text())
        assert summary['training_jobs'] == 4 and summary['evaluation_jobs'] == 8
        assert not any(p.name == 'data' or p.name.startswith('start_') for p in (study/'stages').iterdir())
        for item in summary['evaluations']:
            result = json.loads(verify_reference(item['summary']).read_text())
            assert result['test_objects_read'] == 0
        # Reuse the larger existing fixture to permit a controlled mid-training stop.
        previous = Path(prior['training_stage_stop_resume'])
        stop_plan = json.loads((previous/'plan.json').read_text())
        stop_raw = next((previous/'stages/data').glob('attempt_*/runs/*/dataset.json'))
        stop_cache = next((previous/'stages/start_tv').glob('attempt_*/runs/*/dataset.json'))
        stop_plan.update(selected_starts=['tv'], external_inputs={
            'raw_dataset': file_reference(stop_raw), 'start_caches': {'tv': file_reference(stop_cache)}})
        sp = archive.path/'stop_plan.json'; write_json(sp, stop_plan)
        stopped = archive.path/'stop_resume'
        command('start', '--config', sp, '--root', stopped)
        until = time.monotonic()+120
        while time.monotonic() < until:
            if list((stopped/'stages/train_tv_DP_TWO').glob('attempt_*/runs/*/best_mse.json')): break
            time.sleep(.02)
        else: raise TimeoutError('No checkpoint')
        command('stop', '--root', stopped)
        status = state(stopped, {'interrupted', 'failed', 'completed'})
        assert status['status'] == 'interrupted', status
        time.sleep(.5)
        command('resume', '--root', stopped)
        status = state(stopped, {'completed', 'failed'}); assert status['status'] == 'completed', status
        manifests = list((stopped/'stages/train_tv_DP_TWO').glob('attempt_*/runs/*/manifest.json'))
        assert any(json.loads(p.read_text())['configuration']['execution']['resume'] for p in manifests)
        # All caches are immutable; no new solver/data stages exist after resume either.
        assert not any(p.name == 'data' or p.name.startswith('start_') for p in (stopped/'stages').iterdir())
        after = {str(p): file_reference(p) for p in cache.parent.rglob('*') if p.is_file()}
        assert before == after
        write_json(archive.path/'summary.json', {'passed': True, 'existing_regression': str(regression),
            'continuation': str(study), 'checkpoint_stop_resume': str(stopped), 'guards': rejected,
            'cache_unchanged': True, 'solver_recomputation': False, 'scientific_test_evaluated': False,
            'training_jobs': 4, 'evaluation_jobs': 8})
    return archive.path


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    print(run(p.parse_args().root))
