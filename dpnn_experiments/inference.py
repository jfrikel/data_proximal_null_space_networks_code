# SPDX-License-Identifier: GPL-3.0-only
"""Evaluate published weights on bare artifacts, outside the original study archives.

A JSON specification names the fixed projection (block factorization basis and
sparse operator, or a small fixture NPZ), the recorded projection contract, the
weight files and the sample NPZ files. SHA-256 values are enforced wherever the
specification or the contract provides them, and every checkpoint must carry the
identical projection contract. Rows use the evaluator's metric definitions, so
they are comparable with results/data/test/individual_values.csv.gz. No training,
model selection or test-protocol freezing happens here.
"""
import argparse
import csv
import gzip
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from dpnn_experiments.evaluator import METRICS, error_decomposition, measurement_metrics, summarize
from dpnn_experiments.network_layers import FixedBlockProjection
from dpnn_experiments.operators import Geometry, SparseRadon
from dpnn_experiments.projections import MatrixOperator
from dpnn_experiments.provenance import digest, write_json
from dpnn_experiments.training import build_correction_model

METHODS = ('RES', 'NSN', 'DP', 'DP_TWO')
DEFAULT_TOLERANCES = {'atol': 1e-12, 'rtol': 1e-10}
DEFAULT_MODEL = {'architecture': 'unet', 'width': None, 'dropout': 0.0}
COMPARED_KEYS = ('mse', 'psnr', 'ssim', 'data_change', 'data_residual', 'null_leak',
                 'error_null_mse', 'error_range_mse')


def resolve(root, value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else (Path(root) / path).resolve()


def checked_file(root, ref, expected=None):
    """Accept a path string or {path, sha256?, bytes?}; verify whatever is pinned."""
    ref = {'path': ref} if isinstance(ref, str) else dict(ref)
    path = resolve(root, ref['path'])
    if not path.is_file():
        raise FileNotFoundError(str(path))
    if ref.get('bytes') is not None and path.stat().st_size != ref['bytes']:
        raise ValueError('Artifact size differs from the pinned value: ' + str(path))
    pinned = {x for x in (ref.get('sha256'), expected) if x}
    if len(pinned) > 1:
        raise ValueError('Conflicting pinned hashes for ' + str(path))
    actual = digest(path)
    if pinned and actual not in pinned:
        raise ValueError('Artifact integrity failure: ' + str(path))
    return path, actual


def load_sparse_radon(path, geometry):
    with np.load(path, allow_pickle=False) as arrays:
        radon = SparseRadon(geometry, arrays['rows'], arrays['cols'], arrays['values'])
        if tuple(int(x) for x in arrays['shape']) != radon.shape:
            raise ValueError('Operator shape differs from the contract geometry')
    if not (radon.rows.shape == radon.cols.shape == radon.values.shape) or radon.values.ndim != 1:
        raise ValueError('Inconsistent sparse operator arrays')
    if not np.isfinite(radon.values).all() or radon.rows.min() < 0 or radon.cols.min() < 0 \
            or radon.rows.max() >= radon.shape[0] or radon.cols.max() >= radon.shape[1]:
        raise ValueError('Invalid sparse operator coefficients or indices')
    return radon


def build_projection(spec, root, device):
    p = spec['projection']
    contract = p['contract']
    if isinstance(contract, str):
        contract = json.loads(resolve(root, contract).read_text())
    geometry = contract['geometry']
    if spec.get('geometry') is not None and spec['geometry'] != geometry:
        raise ValueError('Specification geometry differs from the projection contract')
    artifacts = {}
    if p['kind'] == 'block':
        operator_path, artifacts['operator_sha256'] = checked_file(root, p['operator'], contract.get('operator_sha256'))
        basis_path, artifacts['basis_sha256'] = checked_file(root, p['basis'], contract.get('basis_sha256'))
        radon = load_sparse_radon(operator_path, Geometry(**geometry))
        operator = MatrixOperator.whitened_radon(radon, device)
        loaded = np.load(basis_path, allow_pickle=False)
        if loaded.dtype != np.float64 or loaded.ndim != 2:
            raise ValueError('Basis file must contain a two-dimensional Float64 array')
        projection = FixedBlockProjection(torch.from_numpy(loaded).to(device), operator.matrix,
                                          coordinates=contract['coordinates'],
                                          image_weight=(2 / radon.geometry.size) ** 2, contract=contract)
    elif p['kind'] == 'fixture':
        path, artifacts['fixture_sha256'] = checked_file(root, p['reference'], contract.get('sha256'))
        with np.load(path, allow_pickle=False) as d:
            projection = FixedBlockProjection(torch.tensor(d['basis'], device=device),
                                              torch.tensor(d['matrix'], device=device),
                                              coordinates=contract['coordinates'],
                                              image_weight=float(d['image_weight']), contract=contract)
    else:
        raise ValueError('Unknown projection kind')
    if projection.contract != contract:
        differing = sorted(k for k in set(projection.contract) | set(contract)
                           if projection.contract.get(k) != contract.get(k))
        raise ValueError('Reconstructed projection contract differs from the recorded contract: ' + str(differing))
    return projection, artifacts


def build_models(spec, root, projection, device):
    base = {**DEFAULT_MODEL, **spec.get('model', {})}
    models = []
    for entry in spec['models']:
        if entry['method'] not in METHODS:
            raise ValueError('Unknown model method: ' + str(entry.get('method')))
        net = build_correction_model({'model': {**base, 'method': entry['method']}}, projection, device).eval()
        path, sha256 = checked_file(root, entry['weights'])
        state = torch.load(path, map_location=device, weights_only=True)
        net.load_state_dict(state, strict=True)  # enforces the projection contract
        del state
        label = entry.get('label') or '_'.join(str(entry[k]) for k in ('start_kind', 'method', 'seed_index', 'selection') if entry.get(k) is not None)
        models.append({'entry': entry, 'net': net, 'label': label, 'weights_sha256': sha256,
                       'parameter_count': sum(x.numel() for x in net.correction.parameters())})
    if len({m['label'] for m in models}) != len(models):
        raise ValueError('Model labels must be unique')
    return models


def sample_list(spec, root):
    items = spec['samples'] if isinstance(spec['samples'], list) else [spec['samples']]
    samples = []
    for item in items:
        if 'index' in item:
            index = json.loads(resolve(root, item['index']).read_text())
            directory = resolve(root, item['directory'])
            wanted = set(item['ids']) if item.get('ids') is not None else None
            for r in index['records']:
                if item.get('split') and r['split'] != item['split']:
                    continue
                if wanted is not None and r['id'] not in wanted:
                    continue
                samples.append({'id': r['id'], 'path': str(directory / r['file']['name']),
                                'sha256': r['file'].get('sha256'), 'bytes': r['file'].get('bytes'),
                                'start_kind': index.get('start_kind'),
                                'noise_realization_id': r.get('noise_realization_id')})
        else:
            samples.append({'id': item['id'], 'path': item['path'], 'sha256': item.get('sha256'),
                            'bytes': item.get('bytes'), 'start_kind': item.get('start_kind'),
                            'noise_realization_id': item.get('noise_realization_id')})
    if not samples:
        raise ValueError('No samples selected')
    if len({(s['id'], s['start_kind']) for s in samples}) != len(samples):
        raise ValueError('Duplicate sample IDs for the same start kind')
    return samples


def load_sample(root, sample, projection, size, device):
    path, _ = checked_file(root, {k: sample[k] for k in ('path', 'sha256', 'bytes') if sample.get(k) is not None})
    with np.load(path, allow_pickle=False) as z:
        missing = {'start', 'target', 'measurement', 'delta'} - set(z.files)
        if missing:
            raise ValueError(f'Sample {path} lacks {sorted(missing)}')
        data = {k: torch.tensor(z[k], device=device, dtype=torch.float64) for k in z.files
                if k in ('start', 'target', 'measurement', 'analytic_clean', 'raster_clean', 'delta', 'beta')}
    if data['start'].shape != (1, size, size) or data['target'].shape != (1, size, size):
        raise ValueError('Start/target must have shape (1, size, size): ' + str(path))
    if data['measurement'].shape != (projection.matrix.shape[0],):
        raise ValueError('Measurement length differs from the operator: ' + str(path))
    if not all(bool(torch.isfinite(v).all()) for v in data.values()):
        raise ValueError('Nonfinite sample: ' + str(path))
    delta = float(data['delta'])
    if delta < 0:
        raise ValueError('Negative noise norm: ' + str(path))
    start, target, y = data['start'][None], data['target'][None], data['measurement']
    raster = projection.data(target)[0]
    if 'raster_clean' in data and not torch.allclose(raster, data['raster_clean'], atol=1e-10, rtol=1e-10):
        raise ValueError('Raster/operator data mismatch: ' + str(path))
    clean = data.get('analytic_clean')
    if clean is not None and not math.isclose(float((y - clean).norm()), delta, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError('Noise metadata mismatch: ' + str(path))
    return start, target, y, clean, raster, delta, (float(data['beta']) if 'beta' in data else None)


def settings_from(spec, size):
    metrics = {**METRICS, **spec.get('metrics', {})}
    if set(metrics) != set(METRICS):
        raise ValueError('Unknown metric settings: ' + str(sorted(set(metrics) ^ set(METRICS))))
    if metrics['ssim_window'] % 2 != 1 or not 1 <= metrics['ssim_window'] <= size:
        raise ValueError('SSIM window must be odd and fit the image')
    tolerances = {**DEFAULT_TOLERANCES, **spec.get('tolerances', {})}
    if set(tolerances) != {'atol', 'rtol'} or any(not math.isfinite(v) or v < 0 for v in tolerances.values()):
        raise ValueError('Tolerances require finite nonnegative atol and rtol')
    multipliers = spec.get('beta_multipliers', [1.0])
    if not multipliers or len(set(multipliers)) != len(multipliers) or any(not math.isfinite(m) or m < 0 for m in multipliers):
        raise ValueError('Invalid beta multipliers')
    return metrics, tolerances, multipliers


def execute(spec, output, root, *, save_predictions=False, spec_sha256=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'metrics.jsonl').exists():
        raise FileExistsError('Output directory already holds an evaluation: ' + str(output))
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if spec.get('cpu_threads') is not None:
        torch.set_num_threads(int(spec['cpu_threads']))
    device = torch.device(spec.get('device', 'cpu'))
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('Requested CUDA is unavailable; no CPU fallback')
        torch.cuda.set_device(device)
    projection, artifacts = build_projection(spec, root, device)
    size = projection.contract['geometry']['size']
    metrics, tolerances, multipliers = settings_from(spec, size)
    models = build_models(spec, root, projection, device)
    samples = sample_list(spec, root)
    decompose = spec.get('error_decomposition', True)
    rows = []
    with torch.no_grad():
        for index, sample in enumerate(samples):
            start, target, y, clean, raster, delta, cached_beta = load_sample(root, sample, projection, size, device)
            conditions = [('BASE', None, None)] if spec.get('include_baseline', True) else []
            for model in models:
                if sample['start_kind'] and model['entry'].get('start_kind') and sample['start_kind'] != model['entry']['start_kind']:
                    continue
                method = model['entry']['method']
                conditions.extend((method, model, mult) for mult in (multipliers if method in ('DP', 'DP_TWO') else [None]))
            for method, model, mult in conditions:
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                began = time.perf_counter()
                value = null = ran = torch.zeros_like(start)
                t = 0.
                beta = 0. if method in ('BASE', 'NSN') else None
                if model is None:
                    out, details = start, None
                elif method in ('DP', 'DP_TWO'):
                    beta = mult * delta
                    out, details = model['net'](start, beta, return_details=True)
                else:
                    out, details = model['net'](start, return_details=True)
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                seconds = time.perf_counter() - began
                if details is not None:
                    value = details['value']
                    null, ran = projection.components(value) if method == 'RES' else (details['null'], details['range'])
                    t = float(details['control'][0]) if method in ('DP', 'DP_TWO') else details['control']
                row, _ = measurement_metrics(projection, out, start, target, y, raster if clean is None else clean, raster,
                                             delta, beta, (value, null, ran, t), metrics, tolerances, separate=method == 'DP_TWO')
                if clean is None:
                    row['model_error'] = None
                if decompose:
                    row.update(error_decomposition(projection, out, target, tolerances)[0])
                label = model['label'] if model else 'BASE'
                condition = f"{sample['start_kind'] or 'start'}_{method}" + (f'_beta{mult:g}' if mult is not None else '')
                row.update(id=sample['id'], noise_realization_id=sample['noise_realization_id'], method=method,
                           start_kind=sample['start_kind'], condition=condition, model=label,
                           seed_index=model['entry'].get('seed_index') if model else None,
                           selection=model['entry'].get('selection') if model else None,
                           beta_multiplier=mult, cached_beta=cached_beta,
                           weights_sha256=model['weights_sha256'] if model else None,
                           parameter_count=model['parameter_count'] if model else 0,
                           inference_seconds=seconds, bound_formula='beta+norm(APU)' if method == 'DP_TWO' else 'beta+(1-t)*norm(APU)',
                           reference_budget_excess=row['data_change'] - delta)
                rows.append(row)
                with (output / 'metrics.jsonl').open('a', encoding='utf-8') as f:
                    f.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
                if save_predictions:
                    path = output / 'predictions' / f'{index:05d}_{condition}_{label}.npz'
                    path.parent.mkdir(exist_ok=True)
                    np.savez_compressed(path, prediction=out.detach().cpu().numpy(), start=start.cpu().numpy(), target=target.cpu().numpy())
            print(f'Evaluated {sample["id"]} ({index + 1}/{len(samples)})', flush=True)
    with (output / 'metrics.csv').open('x', newline='') as f:
        fields = sorted(set().union(*(r.keys() for r in rows)))
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in r.items()} for r in rows)
    violations = sum(r['constraint_violation'] is True or r.get('error_decomposition_passed') is False for r in rows)
    summary = {'passed': violations == 0, 'violations': violations, 'objects': len(samples), 'rows': len(rows),
               'groups': summarize(rows), 'models': [{'label': m['label'], 'method': m['entry']['method'],
                                                     'start_kind': m['entry'].get('start_kind'), 'seed_index': m['entry'].get('seed_index'),
                                                     'selection': m['entry'].get('selection'), 'weights_sha256': m['weights_sha256'],
                                                     'parameter_count': m['parameter_count']} for m in models],
               'artifacts': artifacts, 'projection_contract': projection.contract, 'metrics': metrics,
               'tolerances': tolerances, 'beta_multipliers': multipliers, 'specification_sha256': spec_sha256,
               'device': str(device), 'torch': str(torch.__version__), 'numpy': np.__version__,
               'definitions': 'measurement_metrics and error_decomposition of dpnn_experiments.evaluator; beta = multiplier * delta',
               'operation': 'inference and metrics only; no training, selection or protocol freezing'}
    write_json(output / 'summary.json', summary)
    return summary


def parse_export_value(text):
    if text in ('True', 'False'):
        return text == 'True'
    try:
        return json.loads(text)
    except ValueError:
        return text or None


def compare_with_export(rows, export_root, *, rtol=1e-8, atol=1e-12):
    """Compare per-object rows with the published test metrics, matched by object, model, start, seed and selection."""
    reference = {}
    with gzip.open(Path(export_root) / 'data' / 'test' / 'individual_values.csv.gz', 'rt') as f:
        for r in csv.DictReader(f):
            parsed = {k: parse_export_value(v) for k, v in r.items()}
            reference[(parsed['id'], parsed['method'], parsed['start_kind'], parsed['seed_index'], parsed['selection'])] = parsed
    compared, missing, worst = 0, [], {k: {'absolute': 0., 'relative': 0.} for k in COMPARED_KEYS}
    failures = []
    for row in rows:
        key = (row['id'], row['method'], row['start_kind'], row['seed_index'], row['selection'])
        if key not in reference:
            missing.append(list(key))
            continue
        compared += 1
        for k in COMPARED_KEYS:
            a, b = row.get(k), reference[key].get(k)
            if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
                continue
            absolute = abs(a - b)
            relative = absolute / abs(b) if b else (0. if absolute == 0 else math.inf)
            worst[k]['absolute'] = max(worst[k]['absolute'], absolute)
            worst[k]['relative'] = max(worst[k]['relative'], relative)
            if absolute > atol + rtol * abs(b):
                failures.append({'key': list(key), 'metric': k, 'computed': a, 'published': b})
    return {'compared_rows': compared, 'missing_in_export': missing, 'max_deviation': worst, 'rtol': rtol, 'atol': atol,
            'failures': failures, 'passed': compared > 0 and not failures}


def make_index(dataset_path, output, *, start_kind=None, copy_to=None):
    """Write a sanitized sample index from an archive dataset.json; optionally copy the sample files."""
    d = json.loads(Path(dataset_path).read_text())
    kind = start_kind or (d.get('start_reconstruction') or {}).get('kind')
    if kind is None and 'tikhonov' in (d.get('configuration') or {}).get('computation', {}):
        kind = 'tikhonov'
    if kind not in ('tikhonov', 'tv'):
        raise ValueError('Specify the start kind (tikhonov or tv)')
    records, names = [], set()
    for r in d['records']:
        source = Path(r['file']['path'])
        if source.name in names:
            raise ValueError('Duplicate sample file name: ' + source.name)
        names.add(source.name)
        if copy_to is not None:
            if source.stat().st_size != r['file']['bytes'] or digest(source) != r['file']['sha256']:
                raise ValueError('Source sample failed its recorded hash: ' + str(source))
            destination = Path(copy_to) / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise FileExistsError(str(destination))
            shutil.copyfile(source, destination)
            if digest(destination) != r['file']['sha256']:
                raise ValueError('Copy changed content: ' + source.name)
        records.append({'id': r['id'], 'split': r['split'], 'noise_realization_id': r.get('noise_realization_id'),
                        'delta': r.get('delta'), 'beta': r.get('beta'),
                        'file': {'name': source.name, 'sha256': r['file']['sha256'], 'bytes': r['file']['bytes']}})
    index = {'schema': 1, 'description': 'Sanitized sample index for dpnn_experiments.inference; file names are relative to the samples directory',
             'start_kind': kind, 'geometry': d['geometry'], 'operator_sha256': d.get('operator_sha256'),
             'start_reconstruction': d.get('start_reconstruction'), 'split_ids': d['split_ids'], 'records': records}
    write_json(output, index)
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run', help='Evaluate the specified weights on the specified samples')
    run.add_argument('--spec', type=Path, required=True)
    run.add_argument('--output', type=Path, required=True)
    run.add_argument('--compare-export', type=Path, help='Results root holding data/test/individual_values.csv.gz')
    run.add_argument('--save-predictions', action='store_true')
    index = sub.add_parser('make-index', help='Write a sanitized sample index from an archive dataset.json')
    index.add_argument('--dataset', type=Path, required=True)
    index.add_argument('--output', type=Path, required=True)
    index.add_argument('--start-kind', choices=['tikhonov', 'tv'])
    index.add_argument('--copy-to', type=Path, help='Copy the sample files into this directory')
    a = parser.parse_args()
    if a.command == 'make-index':
        index = make_index(a.dataset, a.output, start_kind=a.start_kind, copy_to=a.copy_to)
        print(json.dumps({'output': str(a.output), 'records': len(index['records']), 'start_kind': index['start_kind']}, indent=2))
        return
    spec = json.loads(a.spec.read_text())
    summary = execute(spec, a.output, a.spec.resolve().parent, save_predictions=a.save_predictions, spec_sha256=digest(a.spec))
    print(json.dumps({'passed': summary['passed'], 'violations': summary['violations'], 'rows': summary['rows']}, indent=2))
    if a.compare_export is not None:
        rows = [json.loads(line) for line in (a.output / 'metrics.jsonl').read_text().splitlines()]
        comparison = compare_with_export(rows, a.compare_export)
        write_json(a.output / 'comparison.json', comparison)
        print(json.dumps({k: comparison[k] for k in ('compared_rows', 'passed', 'max_deviation')}, indent=2))
        if not comparison['passed']:
            raise SystemExit('Computed metrics deviate from the published export; see comparison.json')


if __name__ == '__main__':
    main()
