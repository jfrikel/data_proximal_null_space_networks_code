# SPDX-License-Identifier: GPL-3.0-only
"""Single-writer input preparation recovery without changing solver settings."""
import copy
from pathlib import Path

from dpnn_experiments.main_integrity import SealedArchive, read
from dpnn_experiments.start_recovery import recovery_candidates
from dpnn_experiments.training_archive import file_reference, verify_reference
from dpnn_experiments.provenance import write_json


def normalized(config):
    c = copy.deepcopy(config)
    for key in ('training', 'preparation_checkpoint_schema', 'reuse_completions', 'stop_after_batches'):
        c.pop(key, None)
    c.get('dataset', {}).pop('reference_operator', None)
    return c


def prepare_stage(kind, config, folder, result, stage, audit, operator_root=None):
    folder = Path(folder); result = Path(result)
    folder.mkdir(parents=True, exist_ok=True)
    intent = {'kind': kind, 'config': config, 'operator_root': operator_root}
    if (folder/'intent.json').exists():
        if read(folder/'intent.json') != intent:
            raise ValueError('Preparation intent changed; use a new study directory')
    else:
        write_json(folder/'intent.json', intent)
    def completed(run):
        archive = SealedArchive(run)
        if normalized(archive.configuration) != normalized(config):
            raise ValueError('Completed preparation differs from intended configuration')
        archive.member('dataset.json'); audit(run)
        return run
    if result.exists():
        previous = read(result)
        run = completed(Path(previous['run']))
        if previous['dataset'] != file_reference(run/'dataset.json'):
            raise ValueError('Preparation result dataset binding changed')
        return run
    partial = []
    for marker in sorted((folder/'runs').glob('*/completion.json'), reverse=True):
        status = read(marker).get('status')
        if status == 'completed':
            run = completed(marker.parent)
            write_json(result, {'run': str(run), 'dataset': file_reference(run/'dataset.json')})
            return run
        if kind in ('start','data') and status in ('failed', 'interrupted'):
            archive = SealedArchive(marker.parent, statuses=('failed', 'interrupted'))
            if normalized(archive.configuration) != normalized(config):
                raise ValueError('Partial preparation differs from intended configuration')
            partial.append(file_reference(marker))
    actual = copy.deepcopy(config)
    if partial and kind=='data':
        # Every continued archive owns a complete copied prefix. Pick one,
        # retaining the other failed attempts without duplicating batch IDs.
        candidates=[(ref,SealedArchive(verify_reference(ref).parent,statuses=('failed','interrupted'))) for ref in partial]
        chosen=max(candidates,key=lambda row:sum(k.startswith('batches/') for k in row[1].entries))
        if any(k.startswith('batches/') for k in chosen[1].entries):actual['reuse_completions']=[chosen[0]]
    elif partial:
        binding = read(verify_reference(config['dataset']['reference']))
        # Each automatic attempt traverses the same ordered full population.
        # Validate every candidate archive separately, then use the longest
        # qualified prefix. Passing all archives would duplicate reused objects.
        candidates = []
        for ref in partial:
            probe = {**config, 'reuse_completions': [ref]}
            candidates.append((ref, set(recovery_candidates(probe, binding))))
        ref, ids = max(candidates, key=lambda row: len(row[1]))
        if any(not other <= ids for _, other in candidates):
            raise ValueError('Non-nested recovery objects require explicit reconciliation')
        if ids:
            actual['reuse_completions'] = [ref]
    run = stage(kind, actual, folder, operator_root) if kind == 'data' else stage(kind, actual, folder)
    completed(run)
    write_json(result, {'run': str(run), 'dataset': file_reference(run/'dataset.json')})
    return run
