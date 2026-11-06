# SPDX-License-Identifier: GPL-3.0-only
"""Manifested real small-CT cache/evaluator and periodic-retention integration tests."""
import argparse,copy,json
from pathlib import Path
import numpy as np
import torch
from tests.start_qualification import run as prepare_fixture
from dpnn_experiments.training import execute as train
from dpnn_experiments.training_archive import TrainingArchive, file_reference, audit_runs
from dpnn_experiments.provenance import write_json
from dpnn_experiments.gpu_benchmark import load_operator, reference_metadata
from dpnn_experiments.projections import MatrixOperator
from dpnn_experiments import evaluator

SOURCE=Path(__file__).resolve().parents[1]

def run(root):
    root=Path(root)
    cfg={'schema_version':1,'purpose':'Cached starts and periodic retention qualification; technical objects only',
         'dataset':{'kind':'four independent technical CT phantoms'},'geometry':{'size':8},
         'randomness':{'seed':4701},'computation':{'device':'cpu','dtype':'float64'}}
    with TrainingArchive(cfg,root/'runs',SOURCE,root/'metadata') as a:
        prep=prepare_fixture(root);summary=json.loads((prep/'summary.json').read_text())
        data=root/'runs'/summary['data_run']/'dataset.json';binding=json.loads(data.read_text())
        operator=root/'operators'/summary['operator_run']
        radon=load_operator(operator,reference_metadata(operator));op=MatrixOperator.whitened_radon(radon,'cpu')
        matrix=op.matrix.to_dense().numpy();_,s,v=np.linalg.svd(matrix,full_matrices=False)
        basis=v[s>1e-12*s[0]].T;projection=a.path/'projection.npz'
        np.savez_compressed(projection,matrix=matrix,basis=basis,image_weight=(2/8)**2)
        caches={x['kind']:root/'runs'/x['run']/'dataset.json' for x in summary['start_runs']}
        starts=[{**json.loads(path.read_text())['start_reconstruction'],'cache':file_reference(path)} for path in caches.values()]
        base=json.loads((SOURCE/'configs/tests/scalable_training_qualification.json').read_text())
        base['geometry']=binding['geometry'];base['projection']={'kind':'fixture','reference':file_reference(projection)}
        base['model'].update(method='DP_TWO',dropout=0.)
        base['dataset']['used_ids']={k:binding['split_ids'][k] for k in ('train','validation')}
        entries=[]
        for kind,path in caches.items():
            tc=copy.deepcopy(base);tc['dataset']['reference']=file_reference(path)
            tr=train(tc,root);best=json.loads((tr/'best_mse.json').read_text())
            entries.append({'method':'DP_TWO','trained_start':kind,'weights':best['weights'],
                            'training_manifest':file_reference(tr/'manifest.json')})
        ec={'schema_version':1,'purpose':'Cached start evaluation; independently generated technical CT objects',
            'dataset':{'reference':file_reference(data)},'geometry':binding['geometry'],
            'randomness':{'seed':4701},'computation':{'device':'cpu','dtype':'float64','cpu_threads':4},
            'model_config':base,'models':entries,'starts':starts,'beta_multipliers':[0.,1.],
            'split':'validation','ids':binding['split_ids']['validation'],'diagnostic_ids':binding['split_ids']['validation'],
            'allow_start_transfer':False,'metrics':{**evaluator.METRICS,'ssim_window':7},
            'tolerances':{'atol':1e-12,'rtol':1e-10},'frozen_protocol':None}
        # Any silent recomputation makes the test fail, even if numerically close.
        original=evaluator.reconstruct
        def forbidden(*args,**kwargs):raise AssertionError('Cache path called reconstruction solver')
        evaluator.reconstruct=forbidden
        try:
            first=evaluator.execute(ec,root);second=evaluator.execute(ec,root)
            one=[json.loads(x) for x in (first/'metrics.jsonl').read_text().splitlines()]
            two=[json.loads(x) for x in (second/'metrics.jsonl').read_text().splitlines()]
            for x,y in zip(one,two):
                for key in x:
                    if isinstance(x[key],(int,float)) and 'seconds' not in key:assert x[key]==y[key],key
            assert len(one)==len(two)==6
            test=copy.deepcopy(ec);test.update(split='test',ids=binding['split_ids']['test'],diagnostic_ids=[])
            lock=a.path/'test_protocol.json';write_json(lock,{'protocol':evaluator.protocol(test),'selection_closed':True})
            test['frozen_protocol']=file_reference(lock);test_run=evaluator.execute(test,root)
            guards=[]
            def reject(label,bad):
                prior=set((root/'runs').iterdir())
                try:evaluator.execute(bad,root)
                except (ValueError,ArithmeticError) as e:
                    guards.append({'case':label,'error':str(e),'runs':[p.name for p in set((root/'runs').iterdir())-prior]})
                else:raise AssertionError('Guard accepted '+label)
            for label in ('settings','hash','training_cache','locked_cache'):
                bad=copy.deepcopy(test if label=='locked_cache' else ec)
                if label=='settings':bad['starts'][0]['alpha']*=2
                elif label=='hash':bad['starts'][0]['cache']['sha256']='0'*64
                elif label=='training_cache':bad['models'][0]['trained_start']='tv';bad['models']=bad['models'][:1];bad['starts']=bad['starts'][1:]
                else:bad['starts'][0]['cache']=file_reference(caches['tv'])
                reject(label,bad)
            # Completed but semantically invalid cache fixtures: hashes alone cannot prove identity.
            for label in ('measurements','solver','split','operator'):
                with TrainingArchive({**cfg,'purpose':'Invalid cache guard fixture: '+label},root/'runs',SOURCE,root/'metadata') as ca:
                    b=json.loads(caches['tikhonov'].read_text());rec=next(r for r in b['records'] if r['id']==ec['ids'][0])
                    if label=='measurements':
                        with np.load(rec['file']['path']) as z:values={k:z[k].copy() for k in z.files}
                        values['measurement'][0]+=.1;sample=ca.path/'altered.npz';np.savez_compressed(sample,**values);rec['file']=file_reference(sample)
                    elif label=='solver':
                        solver=ca.path/'failed_solver.json';write_json(solver,{'passed':False});rec['start_solver']=file_reference(solver)
                    elif label=='split':rec['split']='train'
                    else:b['operator_sha256']='bad'
                    write_json(ca.path/'dataset.json',b)
                bad=copy.deepcopy(ec);bad['models']=[];bad['starts']=bad['starts'][:1];bad['starts'][0]['cache']=file_reference(ca.path/'dataset.json')
                reject(label,bad)
            write_json(a.path/'summary.json',{'passed':True,'validation_runs':[first.name,second.name],
                'test_fixture_run':test_run.name,'identical_repeated_metrics':True,'no_solver_recomputation':True,
                'trained_starts':list(caches),'guards':guards,'scientific_test_used':False})
        finally:evaluator.reconstruct=original
    return a.path

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args();print(run(args.root))
