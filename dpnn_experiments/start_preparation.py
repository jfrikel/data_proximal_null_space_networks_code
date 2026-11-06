"""Immutable start datasets for training: regularized inverse or isotropic TV.

This prepares reconstruction inputs only, never selects parameters from target
quality or evaluates a learned model. Optional isolated subsets retain their original IDs.
"""
import argparse,copy,json,os,shutil,signal
from pathlib import Path
import numpy as np
import torch
from dpnn_experiments.gpu_benchmark import load_operator, reference_metadata
from dpnn_experiments.projections import MatrixOperator
from dpnn_experiments.projection_diagnostics import setup, array
from dpnn_experiments.reconstruction import reconstruct
from dpnn_experiments.training_archive import TrainingArchive, file_reference, verify_reference
from dpnn_experiments.provenance import write_json
from dpnn_experiments.start_recovery import recovery_candidates, selected_records, verified_arrays
SOURCE_ROOT=Path(__file__).resolve().parents[1]


def execute(c,root):
    previous=signal.getsignal(signal.SIGTERM)
    def stop(signum,frame):raise InterruptedError('SIGTERM during start preparation')
    signal.signal(signal.SIGTERM,stop)
    try:return _execute(c,root)
    finally:signal.signal(signal.SIGTERM,previous)


def _execute(c,root):
    root=Path(root);c=copy.deepcopy(c);c['training']={'applicable':False,'reason':'Prepare fixed starts only; no model evaluation or target-based parameter selection'}
    c['preparation_checkpoint_schema']=1
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    with TrainingArchive(c,root/'runs',SOURCE_ROOT,root/'metadata') as a:
        binding=json.loads(verify_reference(c['dataset']['reference']).read_text())
        ref=reference_metadata(c['operator_root'])
        if ref['geometry']!=c['geometry'] or binding['geometry']!=c['geometry'] or ref['files']['operator.npz']!=binding['operator_sha256']:raise ValueError('Inconsistent geometry/operator')
        ids=[x for split in binding['split_ids'].values() for x in split]
        if len(ids)!=len(set(ids)) or set(ids)!={r['id'] for r in binding['records']}:raise ValueError('Invalid dataset split')
        records=selected_records(binding,c)
        candidates=recovery_candidates(c,binding)
        torch.set_num_threads(c['computation']['cpu_threads']);device=setup(a,c)
        op=MatrixOperator.whitened_radon(load_operator(c['operator_root'],ref),device)
        result=copy.deepcopy(binding);result['records']=[];result['source_run']=a.run_id
        result['start_reconstruction']=copy.deepcopy(c['start']);result['parent_dataset']=c['dataset']['reference']
        result['test_policy']='Starts generated with fixed parameters; no learned model or target-based quality evaluation'
        result['configuration']=copy.deepcopy(c)
        chosen={r['id'] for r in records}
        result['split_ids']={k:[i for i in v if i in chosen] for k,v in binding['split_ids'].items()}
        reused=0
        stop_after=c.get('stop_after_objects')
        if stop_after is not None and (not isinstance(stop_after,int) or stop_after<1):raise ValueError('Invalid stop_after_objects')
        with torch.no_grad():
            for i,record in enumerate(records):
                with np.load(verify_reference(record['file']),allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
                candidate=candidates.get(record['id'])
                solver_path=a.path/f'solvers/{i:05d}.json'
                path=a.path/f'samples/{i:05d}.npz';path.parent.mkdir(exist_ok=True)
                if candidate:
                    data=verified_arrays(candidate,data,c['geometry']['size'])
                    solver_path.parent.mkdir(exist_ok=True)
                    shutil.copyfile(candidate['solver'],solver_path)
                    shutil.copyfile(candidate['sample'],path)
                    reused+=1
                else:
                    y=torch.tensor(data['measurement'],device=device,dtype=torch.float64)
                    start,info=reconstruct(op,y,c['geometry']['size'],c['start'])
                    write_json(solver_path,info)
                    if not info['passed']:raise ArithmeticError('Start solver failed qualification stop')
                    data['start']=array(start)[None]
                    # Publish sample before its hash-bound object checkpoint.
                    with path.open('xb') as f:
                        np.savez_compressed(f,**data);f.flush();os.fsync(f.fileno())
                # Remove old start-specific diagnostics; keep measurement provenance.
                rec={k:v for k,v in record.items() if k not in ('file','alpha','cg_iterations','cg_normal_residual','start_solver')}
                rec.update(file=file_reference(path),alpha=c['start']['alpha'],start_solver=file_reference(a.path/f'solvers/{i:05d}.json'))
                rec['preparation_origin']={'reused':bool(candidate),'source_completion':candidate['completion'] if candidate else None,
                    'solver_settings':candidate['original_settings'] if candidate else c['start']}
                write_json(a.path/f'objects/{i:05d}.json',{'id':record['id'],'source':record['file'],'record':rec})
                result['records'].append(rec);a.flush_metadata();print('Reused' if candidate else 'Prepared',record['id'],flush=True)
                if stop_after is not None and len(result['records'])>=stop_after:
                    raise InterruptedError('Requested object-boundary interruption')
        write_json(a.path/'dataset.json',result)
        write_json(a.path/'summary.json',{'objects':len(result['records']),'reused_objects':reused,'computed_objects':len(result['records'])-reused,'start':c['start'],'model_inference':False,'target_quality_evaluation':False})
    return a.path

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    x=p.parse_args();execute(json.loads(x.config.read_text()),x.root)
