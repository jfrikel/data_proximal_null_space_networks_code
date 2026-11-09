#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Assemble the public inference artifacts from the original study archives.

Copies the selected checkpoints, the row-space basis, the operator and the start
caches into one directory with the layout expected by
configs/study/inference_paper_template.json, verifying every file against the
hashes recorded in configs/paper/verified-settings.json and in the projection
contract of results/data/metadata/training_runs.json.xz. The directory, or the
zip archives it can produce, is what gets uploaded to the data repository. No
archive is modified and nothing is computed; files that fail their hash abort.
"""
import argparse
import json
import lzma
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dpnn_experiments.inference import make_index  # noqa: E402
from dpnn_experiments.provenance import digest, write_json  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SELECTIONS = ('best_mse', 'best_ssim')


def weight_name(run, selection):
    return f"{run['start']}_{run['method']}_seed{run['seed_index']}_step{run['position'][selection]['step']:08d}.pt"


def find_run_dir(study_root, run_id, cache):
    if run_id not in cache:
        matches = [p for p in Path(study_root).rglob(run_id) if p.is_dir() and (p / 'manifest.json').is_file()]
        if len(matches) != 1:
            raise FileNotFoundError(f'{len(matches)} run directories named {run_id} under {study_root}')
        cache[run_id] = matches[0]
    return cache[run_id]


def locate_weights(run_dir, step, sha256, size):
    weights = run_dir / 'weights'
    candidates = [weights / f'step_{step:08d}.pt', weights / f'inherited_{sha256}.pt']
    candidates += sorted(p for p in weights.glob('*.pt') if p not in candidates and p.stat().st_size == size)
    for path in candidates:
        if path.is_file() and path.stat().st_size == size and digest(path) == sha256:
            return path
    raise FileNotFoundError(f'No weight file with the recorded hash for step {step} in {run_dir}')


def copy_verified(source, destination, sha256, size=None):
    source, destination = Path(source), Path(destination)
    if size is not None and source.stat().st_size != size:
        raise ValueError('Size differs from the record: ' + str(source))
    if digest(source) != sha256:
        raise ValueError('Source failed its recorded hash: ' + str(source))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if digest(destination) == sha256:
            print('exists, verified:', destination.name, flush=True)
            return destination
        raise FileExistsError('Different file already present: ' + str(destination))
    shutil.copyfile(source, destination)
    if digest(destination) != sha256:
        raise ValueError('Copy changed content: ' + str(destination))
    print('copied:', destination.name, flush=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Artifact root to assemble (DPNN_ARTIFACT_ROOT)')
    parser.add_argument('--study-root', type=Path, help='Study root containing the training run directories (for the weights)')
    parser.add_argument('--factor-run', type=Path, help='Block factorization run directory with row_space_basis.npy and reference/operator.npz')
    parser.add_argument('--tikhonov-dataset', type=Path, help='dataset.json of the Tikhonov start cache')
    parser.add_argument('--tv-dataset', type=Path, help='dataset.json of the TV start cache')
    parser.add_argument('--zip-samples', action='store_true', help='Also write samples_<kind>.zip next to the samples directory')
    parser.add_argument('--verified-settings', type=Path, default=ROOT / 'configs/paper/verified-settings.json')
    parser.add_argument('--training-runs', type=Path, default=ROOT / 'results/data/metadata/training_runs.json.xz')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    settings = json.loads(args.verified_settings.read_text())
    with lzma.open(args.training_runs, 'rt') as f:
        runs = json.load(f)
    contract = runs[0]['projection_contract']
    if any(r['projection_contract'] != contract for r in runs):
        raise ValueError('Training runs record different projection contracts')
    manifest = {'schema': 1, 'layout': 'configs/study/inference_paper_template.json', 'projection_contract': contract,
                'weights': [], 'basis': None, 'operator': None, 'samples': {}}
    if args.study_root is not None:
        cache, by_hash = {}, {}
        for run in settings['runs']:
            run_dir = find_run_dir(args.study_root, run['run_id'], cache)
            for selection in SELECTIONS:
                position = run['position'][selection]
                w = position['weights']
                name = weight_name(run, selection)
                if w['sha256'] not in by_hash:
                    source = locate_weights(run_dir, position['step'], w['sha256'], w['bytes'])
                    copy_verified(source, out / 'weights' / name, w['sha256'], w['bytes'])
                    by_hash[w['sha256']] = {'file': 'weights/' + name, 'sha256': w['sha256'], 'bytes': w['bytes'],
                                           'start': run['start'], 'method': run['method'], 'seed_index': run['seed_index'],
                                           'run_id': run['run_id'], 'step': position['step'], 'epoch': position['epoch'],
                                           'selections': [], 'artifact_id': Path(w['path']).name}
                    manifest['weights'].append(by_hash[w['sha256']])
                elif by_hash[w['sha256']]['file'] != 'weights/' + name:
                    raise ValueError('One weight file would need two names: ' + name)
                by_hash[w['sha256']]['selections'].append(selection)
    if args.factor_run is not None:
        basis = copy_verified(args.factor_run / 'row_space_basis.npy', out / 'row_space_basis.npy', contract['basis_sha256'])
        operator = copy_verified(args.factor_run / 'reference' / 'operator.npz', out / 'operator.npz', contract['operator_sha256'])
        manifest['basis'] = {'file': 'row_space_basis.npy', 'sha256': contract['basis_sha256'], 'bytes': basis.stat().st_size}
        manifest['operator'] = {'file': 'operator.npz', 'sha256': contract['operator_sha256'], 'bytes': operator.stat().st_size}
    write_json(out / 'projection_contract.json', contract)
    for kind, dataset in (('tikhonov', args.tikhonov_dataset), ('tv', args.tv_dataset)):
        if dataset is None:
            continue
        directory = out / 'samples' / kind
        index = make_index(dataset, directory / 'samples_index.json', start_kind=kind, copy_to=directory / 'samples')
        entry = {'index': f'samples/{kind}/samples_index.json', 'directory': f'samples/{kind}/samples',
                 'objects': len(index['records']), 'split_counts': {k: len(v) for k, v in index['split_ids'].items()},
                 'bytes': sum(r['file']['bytes'] for r in index['records'])}
        if args.zip_samples:
            archive = shutil.make_archive(str(out / f'samples_{kind}'), 'zip', root_dir=out / 'samples', base_dir=kind)
            entry['zip'] = {'file': Path(archive).name, 'sha256': digest(archive), 'bytes': Path(archive).stat().st_size,
                            'extract_into': 'samples/'}
        manifest['samples'][kind] = entry
        print(f'samples ({kind}): {entry["objects"]} objects', flush=True)
    files = sorted(p for p in out.rglob('*') if p.is_file() and 'samples' not in p.relative_to(out).parts[:1]
                   and p.name != 'release_manifest.json')
    manifest['files'] = [{'file': str(p.relative_to(out)), 'bytes': p.stat().st_size, 'sha256': digest(p)} for p in files]
    manifest['total_bytes'] = sum(f['bytes'] for f in manifest['files']) + sum(s['bytes'] for s in manifest['samples'].values())
    write_json(out / 'release_manifest.json', manifest)
    print(json.dumps({'output': str(out), 'weights': len(manifest['weights']), 'basis': manifest['basis'] is not None,
                      'operator': manifest['operator'] is not None, 'samples': {k: v['objects'] for k, v in manifest['samples'].items()},
                      'total_gib': round(manifest['total_bytes'] / 2 ** 30, 2)}, indent=2))


if __name__ == '__main__':
    main()
