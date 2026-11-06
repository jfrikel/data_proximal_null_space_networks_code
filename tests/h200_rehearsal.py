"""End-to-end fixed-TV, independent-U-Net, periodic-checkpoint rehearsal."""
import argparse,copy,gc,json
from pathlib import Path
import numpy as np
import torch
from dpnn_experiments.training_archive import TrainingArchive, file_reference, verify_reference
from dpnn_experiments.provenance import write_json
from dpnn_experiments.start_preparation import execute as prepare
from dpnn_experiments.training import execute as train
from tests.training_qualification import compare_runs, rows
from dpnn_experiments import evaluator
SOURCE=Path(__file__).resolve().parents[1]


def release():
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()


def run(c,root):
    root=Path(root)
    with TrainingArchive(c,root/'runs',SOURCE,root/'metadata') as a:
        data=json.loads(verify_reference(c['dataset']['reference']).read_text())
        subset=copy.deepcopy(data)
        subset['split_ids']={k:data['split_ids'][k][:n] for k,n in c['subset_counts'].items()}
        for k,n in c['subset_counts'].items():
            if len(subset['split_ids'][k])!=n:raise ValueError('Insufficient development objects')
        ids={x for v in subset['split_ids'].values() for x in v}
        subset['records']=[r for r in data['records'] if r['id'] in ids]
        subset['parent_dataset']=c['dataset']['reference'];subset['purpose']='Technical rehearsal subset; no scientific test objects'
        write_json(a.path/'subset.json',subset)
        sc={k:copy.deepcopy(c[k]) for k in ('schema_version','geometry','randomness','computation','operator_root','start')}
        sc.update(purpose='Fixed TV inputs for H200 training/resume rehearsal',dataset={'reference':file_reference(a.path/'subset.json')})
        cache=prepare(sc,root);release()
        tc=copy.deepcopy(c['training_config']);tc['dataset']['reference']=file_reference(cache/'dataset.json')
        tc['dataset']['used_ids']={k:subset['split_ids'][k] for k in ('train','validation')}
        tc['training']['epochs']=2;tc['logging'].update(weights_every_step=False,checkpoint_every_steps=1000)
        tc['purpose']='DP_TWO trained on cached TV starts; two-epoch technical H200 rehearsal'
        write_json(a.path/'resolved_training.json',tc)
        full=train(tc,root);release()
        part=copy.deepcopy(tc);part['execution']['stop_after_step']=1
        first=train(part,root);release()
        resume=copy.deepcopy(tc);resume['execution']['resume']=json.loads((first/'latest_checkpoint.json').read_text())
        continued=train(resume,root);release()
        comparison=compare_runs(full,first,continued,c['resume_tolerance'])
        write_json(a.path/'resume_comparison.json',comparison)
        evaluations=[]
        original=evaluator.reconstruct
        def forbidden(*args,**kwargs):raise AssertionError('Cache evaluator attempted to recompute TV')
        evaluator.reconstruct=forbidden
        try:
            for name,source,choice in [('full_mse',full,'best_mse'),('full_ssim',full,'best_ssim'),('resumed_mse',continued,'best_mse')]:
                best=json.loads((source/(choice+'.json')).read_text())
                ec={'schema_version':1,'purpose':'H200 cached TV validation '+name+'; no selection/test access',
                    'dataset':{'reference':file_reference(a.path/'subset.json')},'geometry':c['geometry'],
                    'randomness':c['randomness'],'computation':c['computation'],'model_config':tc,
                    'models':[{'method':'DP_TWO','trained_start':'tv','training_manifest':file_reference(source/'manifest.json'),'weights':best['weights']}],
                    'starts':[{**c['start'],'cache':file_reference(cache/'dataset.json')}],
                    'beta_multipliers':[0.,1.],'split':'validation','ids':subset['split_ids']['validation'],
                    'diagnostic_ids':subset['split_ids']['validation'],'allow_start_transfer':False,
                    'metrics':{**evaluator.METRICS,'ssim_window':tc['metrics']['ssim_window']},
                    'tolerances':{'atol':1e-12,'rtol':1e-10},'frozen_protocol':None}
                out=evaluator.execute(ec,root);release();evaluations.append((name,out))
        finally:evaluator.reconstruct=original
        cache_binding=json.loads((cache/'dataset.json').read_text());records={r['id']:r for r in cache_binding['records']}
        for name,path in evaluations:
            for row in rows(path/'metrics.jsonl'):
                with np.load(verify_reference(row['prediction'])) as pred,np.load(verify_reference(records[row['id']]['file'])) as stored:
                    if not np.array_equal(pred['start'][0],stored['start']):raise AssertionError('Start cache changed')
                if row['constraint_violation']:raise AssertionError('Budget violation')
        # Same selection after resume: compare complete predictions, not only means.
        differences=[]
        for x,y in zip(rows(evaluations[0][1]/'metrics.jsonl'),rows(evaluations[2][1]/'metrics.jsonl')):
            assert (x['id'],x['condition'])==(y['id'],y['condition'])
            with np.load(verify_reference(x['prediction'])) as px,np.load(verify_reference(y['prediction'])) as py:
                differences.append(float(np.max(np.abs(px['prediction']-py['prediction']))))
        if max(differences)>c['evaluation_tolerance']:raise AssertionError('Resumed evaluation mismatch')
        result={'passed':True,'cache_run':cache.name,'full_run':full.name,'interrupted_run':first.name,'resumed_run':continued.name,
            'resume_comparison':comparison,'evaluation_runs':{name:path.name for name,path in evaluations},
            'maximum_prediction_difference':max(differences),'cache_arrays_exact':True,'solver_recomputation':False,
            'scientific_test_used':False,'training_ids':subset['split_ids']['train'],'validation_ids':subset['split_ids']['validation'],
            'full_weight_files':len(list((full/'weights').glob('*.pt'))),'full_steps':len(rows(full/'training_steps.jsonl'))}
        write_json(a.path/'summary.json',result);print(json.dumps(result,indent=2),flush=True)
    return a.path

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--root',type=Path,required=True)
    args=ap.parse_args();run(json.loads(args.config.read_text()),args.root)
