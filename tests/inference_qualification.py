# SPDX-License-Identifier: GPL-3.0-only
"""Archived CPU qualification of the public inference entry point on the logger fixture.

Trains each method briefly on the fixture, then checks that the inference entry
point replays the archived validation metrics from the bare weights, basis and
samples, rejects altered weights, samples and contracts, and that the export
comparison detects a perturbed value.
"""
import argparse
import copy
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from dpnn_experiments import inference
from dpnn_experiments.provenance import write_json
from dpnn_experiments.training import execute
from dpnn_experiments.training_archive import TrainingArchive, file_reference, verify_reference
from tests.training_qualification import fixture, rows

SOURCE_ROOT = Path(__file__).resolve().parents[1]
REPLAYED = ('mse', 'psnr', 'ssim', 'data_change', 'data_residual', 'null_leak', 'leakage_bound', 'signed_beta_excess', 'control')


def expect_error(action, kind, message):
    try:
        action()
    except kind as error:
        if message not in str(error):
            raise
        return
    raise AssertionError('Expected rejection did not happen: ' + message)


def run(config, root):
    root = Path(root)
    outer = copy.deepcopy(config)
    outer['purpose'] = 'Inference entry point qualification on the logger fixture: metric replay, hash and contract rejection'
    outer['training'] = {'applicable': False, 'reason': 'Orchestrates archived fixture trainings and inference replays'}
    with TrainingArchive(outer, root / 'runs', SOURCE_ROOT, root / 'metadata') as archive:
        c = copy.deepcopy(config)
        reference, projection, splits, _ = fixture(archive, c)
        c['dataset']['reference'] = reference
        c['dataset']['used_ids'] = {k: splits[k] for k in ('train', 'validation')}
        c['projection'] = projection
        binding = json.loads(verify_reference(reference).read_text())
        validation = [r for r in binding['records'] if r['split'] == 'validation']
        samples = [{'id': r['id'], 'path': r['file']['path'], 'sha256': r['file']['sha256'], 'bytes': r['file']['bytes'],
                    'start_kind': 'tikhonov', 'noise_realization_id': r['noise_realization_id']} for r in validation]
        replays, checks, worst = [], [], 0.
        specs = {}
        for method in ('RES', 'NSN', 'DP', 'DP_TWO'):
            cfg = copy.deepcopy(c)
            cfg['model']['method'] = method
            run_path = execute(cfg, root)
            best = json.loads((run_path / 'best_mse.json').read_text())
            contract = json.loads((run_path / 'projection_contract.json').read_text())
            spec = {'schema': 1, 'device': 'cpu', 'cpu_threads': cfg['computation']['cpu_threads'],
                    'projection': {'kind': 'fixture', 'reference': projection['reference'], 'contract': contract},
                    'model': {k: cfg['model'][k] for k in ('architecture', 'width', 'dropout')},
                    'models': [{'method': method, 'start_kind': 'tikhonov', 'seed_index': 0, 'selection': 'best_mse',
                                'weights': best['weights']}],
                    'samples': samples, 'metrics': {k: cfg['metrics'][k] for k in inference.METRICS},
                    'beta_multipliers': [1.0], 'include_baseline': True, 'error_decomposition': True}
            spec_path = archive.path / f'inference_spec_{method}.json'
            write_json(spec_path, spec)
            specs[method] = (spec, run_path, best)
            summary = inference.execute(spec, archive.path / f'inference_{method}', archive.path)
            if not summary['passed'] or summary['rows'] != 2 * len(samples):
                raise AssertionError('Inference replay incomplete or violated bounds: ' + method)
            computed = [json.loads(x) for x in (archive.path / f'inference_{method}' / 'metrics.jsonl').read_text().splitlines()]
            archived = rows(run_path / 'metrics.jsonl')
            differences = []
            for row in computed:
                expected_hash = None if row['method'] == 'BASE' else best['weights']['sha256']
                match = [r for r in archived if r['id'] == row['id'] and r['weight_sha256'] == expected_hash]
                if len(match) != 1:
                    raise AssertionError('Archived validation row not found for ' + row['id'])
                for key in REPLAYED:
                    a, b = row.get(key), match[0].get(key)
                    # The training logger leaves budget fields empty for unconstrained RES
                    # rows; the evaluator definitions used here record them for every method.
                    if b is None:
                        continue
                    if a is None:
                        raise AssertionError(f'Metric {key} missing in the inference row ({method})')
                    differences.append(abs(a - b))
            worst = max(worst, max(differences))
            replays.append({'method': method, 'training_run': run_path.name, 'rows': len(computed),
                            'max_absolute_difference': max(differences)})
        if worst > 1e-12:
            raise AssertionError('Inference metrics differ from the archived validation metrics: ' + str(worst))
        checks.append('validation metrics replayed from bare weights for RES, NSN, DP and DP_TWO')
        spec, run_path, best = specs['DP']
        # Altered weights with a pinned hash are rejected before deserialization.
        corrupt = archive.path / 'corrupt_weights.pt'
        corrupt.write_bytes(Path(best['weights']['path']).read_bytes() + b'corruption')
        bad = copy.deepcopy(spec)
        bad['models'][0]['weights'] = {**best['weights'], 'path': str(corrupt), 'bytes': corrupt.stat().st_size}
        expect_error(lambda: inference.execute(bad, archive.path / 'rejected_weights', archive.path), ValueError, 'integrity')
        checks.append('altered weights rejected')
        # A contract that differs from the checkpoint is rejected when loading.
        bad = copy.deepcopy(spec)
        bad['projection']['contract']['kind'] = 'other_fixture'
        expect_error(lambda: inference.execute(bad, archive.path / 'rejected_contract', archive.path), ValueError, 'contract mismatch')
        checks.append('projection contract mismatch rejected')
        # A sample whose recorded raster data do not belong to the operator is rejected.
        with np.load(validation[0]['file']['path'], allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        arrays['raster_clean'] = arrays['measurement'] * 1.5
        altered = archive.path / 'altered_sample.npz'
        np.savez_compressed(altered, **arrays)
        bad = copy.deepcopy(spec)
        bad['samples'] = [{**samples[0], **file_reference(altered)}]
        expect_error(lambda: inference.execute(bad, archive.path / 'rejected_sample', archive.path), ValueError, 'Raster/operator')
        checks.append('sample/operator mismatch rejected')
        # The export comparison accepts identical values and detects a perturbation.
        computed = [json.loads(x) for x in (archive.path / 'inference_DP' / 'metrics.jsonl').read_text().splitlines()]
        export = archive.path / 'fake_export' / 'data' / 'test'
        export.mkdir(parents=True)
        fields = ['id', 'method', 'start_kind', 'seed_index', 'selection', *inference.COMPARED_KEYS]
        def write_export(perturb):
            with gzip.open(export / 'individual_values.csv.gz', 'wt', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                for i, r in enumerate(computed):
                    values = {k: ('' if r.get(k) is None else r[k]) for k in fields}
                    if perturb and i == len(computed) - 1:
                        values['mse'] = r['mse'] * (1 + 1e-6)
                    writer.writerow(values)
        write_export(False)
        same = inference.compare_with_export(computed, archive.path / 'fake_export')
        if not same['passed'] or same['compared_rows'] != len(computed) or same['missing_in_export']:
            raise AssertionError('Export comparison failed on identical values')
        (export / 'individual_values.csv.gz').unlink()
        write_export(True)
        perturbed = inference.compare_with_export(computed, archive.path / 'fake_export')
        if perturbed['passed'] or len(perturbed['failures']) != 1:
            raise AssertionError('Export comparison missed a perturbed value')
        checks.append('export comparison accepts identical rows and flags a perturbed value')
        # The sanitized index reproduces the sample set through the index form of the specification.
        index_path = archive.path / 'samples_index.json'
        inference.make_index(verify_reference(reference), index_path, start_kind='tikhonov',
                             copy_to=archive.path / 'copied_samples')
        indexed = copy.deepcopy(spec)
        indexed['samples'] = {'index': str(index_path), 'directory': str(archive.path / 'copied_samples'), 'split': 'validation'}
        summary = inference.execute(indexed, archive.path / 'inference_DP_indexed', archive.path)
        replayed = [json.loads(x) for x in (archive.path / 'inference_DP_indexed' / 'metrics.jsonl').read_text().splitlines()]
        if summary['rows'] != len(computed) or any(a['mse'] != b['mse'] for a, b in zip(replayed, computed)):
            raise AssertionError('Index-based sample selection does not replay the explicit sample list')
        checks.append('sanitized sample index and copied samples replay the explicit list')
        result = {'replays': replays, 'max_absolute_difference': worst, 'checks': checks, 'all_completed': True,
                  'real_256_inference': False}
        write_json(archive.path / 'summary.json', result)
        print(json.dumps(result, indent=2), flush=True)
    return archive.path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    run(json.loads(a.config.read_text()), a.root)


if __name__ == '__main__':
    main()
