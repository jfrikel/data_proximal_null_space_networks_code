"""Reuse only qualified objects from hash-bound, closed preparation archives."""
import json
from pathlib import Path
import numpy as np
from dpnn_experiments.training_archive import verify_reference
from dpnn_experiments.provenance import digest


def selected_records(binding, config):
    records=binding['records']
    ids=config.get('only_ids')
    if ids is None:return records
    if not ids or len(ids)!=len(set(ids)) or not set(ids)<=set(r['id'] for r in records):
        raise ValueError('Invalid preparation subset')
    return [r for r in records if r['id'] in set(ids)]


def recovery_candidates(config, binding):
    candidates={}
    for reference in config.get('reuse_completions',[]):
        completion_path=verify_reference(reference);root=completion_path.parent
        completion=json.loads(completion_path.read_text())
        if completion.get('status') not in ('completed','failed','interrupted'):
            raise ValueError('Recovery requires a closed archive')
        for name,key in [('manifest.json','manifest_sha256'),('artifacts.json','artifacts_sha256')]:
            if digest(root/name)!=completion.get(key):raise ValueError('Recovery archive binding mismatch')
        old=json.loads((root/'manifest.json').read_text())['configuration']
        if old.get('dataset',{}).get('reference')!=config['dataset']['reference']:
            raise ValueError('Recovery input dataset mismatch')
        if old['geometry']!=config['geometry'] or old['computation']!=config['computation']:
            raise ValueError('Recovery geometry/computation mismatch')
        # A larger cap does not change an already converged solution. No other
        # solver setting (in particular tolerances or alpha) may change.
        before=dict(old['start']);after=dict(config['start'])
        old_limit=before.pop('max_iterations');new_limit=after.pop('max_iterations')
        if before!=after or new_limit<old_limit:raise ValueError('Recovery solver settings mismatch')
        entries=json.loads((root/'artifacts.json').read_text())['files']
        index={e['path']:e for e in entries}
        if len(index)!=len(entries):raise ValueError('Duplicate recovery artifact')
        def artifact(name):
            entry=index[name]
            return verify_reference({'path':str(root/name),'bytes':entry['bytes'],'sha256':entry['sha256']})
        if 'manifest.json' not in index or artifact('manifest.json')!=root/'manifest.json':
            raise ValueError('Missing indexed recovery manifest')
        sources={x['path']:x['sha256'] for x in json.loads(artifact('sources.json').read_text())}
        for module in ('reconstruction.py','projections.py','pilot_data.py','gpu_benchmark.py','operators.py'):
            name='dpnn_experiments/'+module
            if sources.get(name)!=digest(Path(__file__).parent/module):
                raise ValueError('Recovery numerical implementation mismatch: '+module)
        for i,record in enumerate(selected_records(binding,old)):
            solver_name=f'solvers/{i:05d}.json';sample_name=f'samples/{i:05d}.npz'
            object_name=f'objects/{i:05d}.json'
            if old.get('preparation_checkpoint_schema')==1 and object_name not in index:
                continue  # sample not yet committed at interruption

            if solver_name not in index:continue
            solver=artifact(solver_name);info=json.loads(solver.read_text())
            if info.get('passed') is not True:continue
            if sample_name not in index:continue  # crash between solver and sample
            sample=artifact(sample_name)
            if record['id'] in candidates:raise ValueError('Overlapping recovery objects')
            original_settings=old['start']
            if old.get('preparation_checkpoint_schema')==1:
                checkpoint=json.loads(artifact(object_name).read_text())
                if checkpoint['id']!=record['id'] or checkpoint['source']!=record['file']:
                    raise ValueError('Object checkpoint input binding mismatch')
                committed=checkpoint['record']
                for key,name in [('file',sample_name),('start_solver',solver_name)]:
                    if any(committed[key][k]!=index[name][k] for k in ('sha256','bytes')):
                        raise ValueError('Object checkpoint output binding mismatch')
                original_settings=committed['preparation_origin']['solver_settings']
            candidates[record['id']]={'sample':sample,'solver':solver,'completion':reference,
                                      'original_settings':original_settings}
    return candidates


def verified_arrays(candidate, source, size):
    with np.load(candidate['sample'],allow_pickle=False) as z:
        arrays={k:z[k].copy() for k in z.files}
    if set(arrays)!=set(source):raise ValueError('Recovery array keys mismatch')
    for key in source:
        if key!='start' and not np.array_equal(arrays[key],source[key]):
            raise ValueError('Recovery source content mismatch: '+key)
    start=arrays['start']
    if start.shape!=(1,size,size) or start.dtype!=np.float64 or not np.isfinite(start).all():
        raise ValueError('Invalid recovered start')
    return arrays
