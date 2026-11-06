"""Archived logger, crash-window, checkpoint/resume and solver qualification."""
import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np
import torch

from dpnn_experiments.pilot_data import tikhonov
from dpnn_experiments.projections import MatrixOperator
from dpnn_experiments.provenance import write_json, digest
from dpnn_experiments.training import execute, validate_config, CachedData, aggregate, update_best
from dpnn_experiments.training_archive import TrainingArchive, atomic_torch_save, file_reference, verify_reference, audit_runs

SOURCE_ROOT=Path(__file__).resolve().parents[1]


def rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines()]


def tensor_difference(a,b):
    if isinstance(a,torch.Tensor):
        if a.shape!=b.shape or a.dtype!=b.dtype:raise AssertionError('Tensor shape/dtype differs')
        return float((a.cpu()-b.cpu()).abs().max()) if a.numel() else 0.
    if isinstance(a,dict):
        if a.keys()!=b.keys():raise AssertionError('State keys differ')
        return max([tensor_difference(a[k],b[k]) for k in a]+[0.])
    if isinstance(a,(tuple,list)):
        if len(a)!=len(b):raise AssertionError('State lengths differ')
        return max([tensor_difference(x,y) for x,y in zip(a,b)]+[0.])
    if a!=b:raise AssertionError(f'State values differ: {a!r} vs {b!r}')
    return 0.


def compare_runs(full,first,resumed,tolerance):
    a=rows(full/'training_steps.jsonl');b=rows(first/'training_steps.jsonl')+rows(resumed/'training_steps.jsonl')
    # A failed segment may contain work after its last committed checkpoint.
    start_state=torch.load(json.loads((resumed/'manifest.json').read_text())['configuration']['execution']['resume']['path'],
                           map_location='cpu',weights_only=True)
    b=[r for r in rows(first/'training_steps.jsonl') if r['step']<=start_state['progress']['step']]+rows(resumed/'training_steps.jsonl')
    if len(a)!=len(b):raise AssertionError('Optimizer step count differs')
    losses=[]
    for x,y in zip(a,b):
        for key in ('step','epoch','ids','learning_rates','beta','delta'):
            if x[key]!=y[key]:raise AssertionError('Batch/parameter replay differs: '+key)
        losses.append(abs(x['loss']-y['loss']))
    ca=json.loads((full/'latest_checkpoint.json').read_text());cb=json.loads((resumed/'latest_checkpoint.json').read_text())
    sa=torch.load(verify_reference(ca),map_location='cpu',weights_only=True)
    sb=torch.load(verify_reference(cb),map_location='cpu',weights_only=True)
    wa=torch.load(verify_reference(sa['weight_reference']),map_location='cpu',weights_only=True)
    wb=torch.load(verify_reference(sb['weight_reference']),map_location='cpu',weights_only=True)
    differences={'weights':tensor_difference(wa,wb),'optimizer':tensor_difference(sa['optimizer'],sb['optimizer']),
                 'scheduler':tensor_difference(sa['scheduler'],sb['scheduler']),
                 'rng':tensor_difference(sa['rng'],sb['rng']),'loss':max(losses,default=0.)}
    # Discrete decisions must replay exactly. Metric-valued state uses the same
    # declared tolerance as weights/losses: CUDA Float64 can differ by an ULP
    # after allocations change on resume, even with deterministic algorithms.
    for key in ('step','epoch','cursor','order','phase','bad_epochs'):
        if sa['progress'][key]!=sb['progress'][key]:raise AssertionError('Resume position mismatch: '+key)
    for key in ('epoch','step'):
        if sa['progress']['best'][key]!=sb['progress']['best'][key]:
            raise AssertionError('Selected best checkpoint position differs: '+key)
    for metric in ('mse','ssim'):
        key='best_'+metric
        for field in ('epoch','step'):
            if sa['progress'][key][field]!=sb['progress'][key][field]:
                raise AssertionError('Parallel checkpoint selection differs: '+key)
        differences[key]=abs(sa['progress'][key][metric]-sb['progress'][key][metric])
        ra=verify_reference(sa['progress'][key]['weights'])
        rb=verify_reference(sb['progress'][key]['weights'])
        differences[key+'_weights']=tensor_difference(torch.load(ra,weights_only=True),torch.load(rb,weights_only=True))
    differences['early_stopping_reference']=abs(sa['progress']['early_stopping_reference']-sb['progress']['early_stopping_reference'])
    if sa['progress']['best']['mse']!=sb['progress']['best']['mse']:
        differences['best_mse']=abs(sa['progress']['best']['mse']-sb['progress']['best']['mse'])
    if max(differences.values())>tolerance:raise AssertionError('Replay exceeded tolerance: '+str(differences))
    return {'full':full.name,'interrupted':first.name,'resumed':resumed.name,'tolerance':tolerance,
            'absolute_differences':differences,'bitwise_tensor_and_loss_match':all(v==0 for v in differences.values())}


def fixture(archive,config):
    rng=np.random.Generator(np.random.PCG64(71523));size=config['geometry']['size'];n=size*size
    u,_=np.linalg.qr(rng.normal(size=(12,6)));v,_=np.linalg.qr(rng.normal(size=(n,6)))
    matrix=(u*np.geomspace(2,.05,6))@v.T;wx=(2/size)**2
    projection=archive.path/'fixture_projection.npz'
    np.savez_compressed(projection,basis=v,matrix=matrix,image_weight=wx)
    operator=MatrixOperator.dense(matrix,'cpu');records=[]
    splits={'train':[f'logger-fixture/train/{i}' for i in range(5)],
            'validation':[f'logger-fixture/validation/{i}' for i in range(3)],
            'test':[f'logger-fixture/test/{i}' for i in range(2)]}
    errors=[]
    for split,names in splits.items():
        for name in names:
            target=rng.uniform(.1,.8,(1,size,size));clean=matrix@target.ravel()*np.sqrt(wx)
            noise=rng.normal(size=12);noise*=.05*np.linalg.norm(clean)/np.linalg.norm(noise)
            y=clean+noise
            x,info=tikhonov(operator,torch.tensor(y[:,None]),alpha=.01,rtol=1e-10,atol=1e-12,max_iterations=100)
            exact=np.linalg.solve(matrix.T@matrix+.01*np.eye(n),matrix.T@y)
            error=float(np.linalg.norm(x.numpy().ravel()-exact));errors.append(error)
            if not info['passed'] or error>1e-9:raise AssertionError('Small Tikhonov reference failed')
            p=archive.path/'samples'/name.replace('/','_');p=p.with_suffix('.npz');p.parent.mkdir(exist_ok=True)
            np.savez_compressed(p,start=x.numpy().reshape(1,size,size)/np.sqrt(wx),target=target,
                                measurement=y,delta=np.linalg.norm(noise),beta=np.linalg.norm(noise))
            records.append({'id':name,'split':split,'file':file_reference(p),'noise_realization_id':name+'/0'})
            write_json(archive.path/(name.replace('/','_')+'_cg.json'),info)
    path=archive.path/'dataset.json'
    write_json(path,{'schema':1,'geometry':config['geometry'],'records':records,'split_ids':splits,
                    'purpose':'Logger qualification only; random rank-six matrix, not CT quality'})
    return file_reference(path),{'kind':'fixture','reference':file_reference(projection)},splits,max(errors)


def run(config,root,real_config=None):
    root=Path(root);outer=copy.deepcopy(config)
    outer['purpose']='A20.1/A20.2 logger qualification: exact replay, failures, metrics and reserved test split'
    outer['training']={'applicable':False,'reason':'Orchestrates separately archived training fixtures'}
    with TrainingArchive(outer,root/'runs',SOURCE_ROOT,root/'metadata') as archive:
        c=copy.deepcopy(real_config or config)
        if real_config is None:
            ref,projection,splits,cg_error=fixture(archive,c)
            c['dataset']['reference']=ref;c['dataset']['used_ids']={k:splits[k] for k in ('train','validation')}
            c['projection']=projection
        else:cg_error=None
        write_json(archive.path/'resolved_training_config.json',c);archive.flush_metadata()
        full=execute(c,root)
        if not json.loads((full/'manifest.json').read_text())['model_evaluation']['applicable']:
            raise AssertionError('Manifest must identify actual model evaluations')
        # Independent selection oracle over archived validation rows (no helper reuse).
        events=rows(full/'evaluations.jsonl')
        candidates=[x for x in events if x['status']=='completed' and x['weight_reference'] is not None]
        for metric in ('mse','ssim'):
            expected=(min if metric=='mse' else max)(candidates,key=lambda x:x['summary'][metric]['mean'])
            selected=json.loads((full/('best_'+metric+'.json')).read_text())
            assert selected['weights']==expected['weight_reference']
            assert selected[metric]==expected['summary'][metric]['mean']
        assert json.loads((full/'best_model.json').read_text())==json.loads((full/'best_mse.json').read_text())
        comparisons=[]
        stops=[1] if real_config else [2,3,4]
        for stop in stops:
            partial=copy.deepcopy(c);partial['execution']['stop_after_step']=stop
            first=execute(partial,root)
            if json.loads((first/'completion.json').read_text())['status']!='interrupted':raise AssertionError('Missing interruption status')
            resumed=copy.deepcopy(c);resumed['execution']['resume']=json.loads((first/'latest_checkpoint.json').read_text())
            last=execute(resumed,root)
            comparisons.append(compare_runs(full,first,last,1e-12))
        checks=[]
        progress={'best':None,'best_mse':None,'best_ssim':None,'bad_epochs':7,'early_stopping_reference':.1}
        sequence=[(.4,.6),(.3,.5),(.35,.8),(.3,.8)]
        for epoch,(mse,ssim_value) in enumerate(sequence):
            progress.update(epoch=epoch,step=epoch+1)
            update_best(progress,{'mse':{'mean':mse},'ssim':{'mean':ssim_value}}, {'fixture_step':epoch+1})
        assert progress['best_mse']['step']==2 and progress['best_ssim']['step']==3
        assert progress['best']==progress['best_mse'] and progress['bad_epochs']==7 and progress['early_stopping_reference']==.1
        before=copy.deepcopy(progress)
        try:update_best(progress,{'mse':{'mean':.1},'ssim':{'mean':float('nan')}},{})
        except ArithmeticError:pass
        else:raise AssertionError('Nonfinite metric accepted')
        assert progress==before
        write_json(archive.path/'dual_selection_fixture.json',{'sequence':sequence,'result':progress})
        checks.append('independent optima, first tie, legacy alias, finite guard and no patience mutation')
        if real_config is None:
            missing=copy.deepcopy(c);del missing['optimizer']['eps']
            write_json(archive.path/'rejected_missing_metadata.json',missing)
            try:validate_config(missing)
            except ValueError:checks.append('missing effective parameter rejected before training')
            else:raise AssertionError('Missing metadata accepted')
            data=CachedData(c)
            try:data.batch(data.manifest['split_ids']['test'][:1],'cpu',split='test')
            except ValueError:checks.append('test data access rejected by training loader')
            else:raise AssertionError('Test leakage')
            safe=archive.path/'atomic_test.pt';atomic_torch_save({'previous':1},safe);before=digest(safe)
            def failing_writer(value,stream):stream.write(b'partial');raise OSError('Injected disk-write error')
            try:atomic_torch_save({'next':2},archive.path/'failed_atomic.pt',writer=failing_writer)
            except OSError:pass
            else:raise AssertionError('Injected write failure not propagated')
            if (archive.path/'failed_atomic.pt').exists() or digest(safe)!=before:raise AssertionError('Atomic checkpoint corruption')
            checks.append('atomic write failure leaves previous artifact intact')
            blocked=archive.path/'not_a_directory';blocked.write_text('fixture')
            try:execute(c,blocked)
            except (NotADirectoryError,FileExistsError):checks.append('unwritable run root rejected before numerical work')
            else:raise AssertionError('Invalid output root accepted')
            incomplete=archive.path/'incomplete_fixture'/'runs'/'unfinished';incomplete.mkdir(parents=True)
            write_json(incomplete/'manifest.json',{'purpose':'Simulated hard-crash fixture, intentionally no completion marker'})
            if audit_runs(incomplete.parent.parent)[0]['status']!='unfinished_or_running':raise AssertionError('Incomplete run misclassified')
            checks.append('missing completion marker classified unfinished, never successful')
            corrupt=archive.path/'corrupt.pt';corrupt.write_bytes(safe.read_bytes()+b'corruption')
            bad_reference={**file_reference(safe),'path':str(corrupt)}
            try:verify_reference(bad_reference)
            except ValueError:checks.append('altered checkpoint rejected before deserialization')
            else:raise AssertionError('Corrupt checkpoint accepted')
            bad_dataset=copy.deepcopy(data.manifest)
            bad_dataset['split_ids']['validation'][0]=bad_dataset['split_ids']['train'][0]
            bad_path=archive.path/'overlapping_dataset.json';write_json(bad_path,bad_dataset)
            overlap=copy.deepcopy(c);overlap['dataset']['reference']=file_reference(bad_path)
            try:CachedData(overlap)
            except ValueError:checks.append('overlapping object splits rejected')
            else:raise AssertionError('Overlapping splits accepted')
            broken=copy.deepcopy(c);broken['execution']['injected_io_failure_step']=2
            prior=set((root/'runs').iterdir())
            try:execute(broken,root)
            except OSError:pass
            else:raise AssertionError('Logging failure did not stop training')
            failed=next(iter(set((root/'runs').iterdir())-prior))
            if json.loads((failed/'completion.json').read_text())['status']!='failed':raise AssertionError('Failure status missing')
            retry=copy.deepcopy(c);retry['execution']['resume']=json.loads((failed/'latest_checkpoint.json').read_text())
            last=execute(retry,root);comparisons.append(compare_runs(full,failed,last,1e-12))
            mismatch=copy.deepcopy(retry);mismatch['optimizer']['lr']*=2
            try:execute(mismatch,root)
            except ValueError as e:
                if 'mismatch' not in str(e):raise
                checks.append('incompatible resume rejected')
            else:raise AssertionError('Changed optimizer accepted')
            evaluation=copy.deepcopy(c);evaluation['mode']='evaluate'
            evaluation['execution']['resume']=json.loads((full/'latest_checkpoint.json').read_text())
            separate=execute(evaluation,root)
            checks.append('standalone validation archived with checkpoint hash: '+separate.name)
            # Complete metrics tables must be derivable from per-object rows alone.
            values=rows(full/'metrics.jsonl');last_id=values[-1]['evaluation_id']
            rebuilt=aggregate([v for v in values if v['evaluation_id']==last_id])
            summary=json.loads((full/f"validation_{values[-1]['step']:08d}_summary.json").read_text())
            if rebuilt!=summary:raise AssertionError('Metrics aggregation does not replay')
            checks.append('summary exactly rebuilt from object rows')
            stopping=copy.deepcopy(c);stopping['optimizer']['lr']=0.;stopping['selection']['patience']=1
            stopped=execute(stopping,root);position=json.loads((stopped/'summary.json').read_text())['position']
            if position['step']!=6 or position['best']['step']!=3:raise AssertionError('Early stopping/best selection failed')
            checks.append('early stopping and retained best validation checkpoint verified')
            margin=copy.deepcopy(c);margin['selection'].update(patience=1,min_delta=1.)
            margin_run=execute(margin,root);position=json.loads((margin_run/'summary.json').read_text())['position']
            if position['step']!=6 or position['best']['step']!=6:
                raise AssertionError('min_delta incorrectly changed the best-checkpoint criterion')
            checks.append('min_delta controls patience independently of the actual minimum-MSE checkpoint')
        result={'comparisons':comparisons,'checks':checks,'tikhonov_reference_max_absolute_error':cg_error,
                'all_completed':True,'real_256_smoke':real_config is not None,'test_evaluated':False}
        write_json(archive.path/'summary.json',result)
        print(json.dumps(result,indent=2),flush=True)
    return archive.path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--real-config',type=Path)
    a=p.parse_args();config=json.loads(a.config.read_text())
    os.environ['CUBLAS_WORKSPACE_CONFIG']=config['computation']['cublas_workspace_config']
    run(config,a.root,json.loads(a.real_config.read_text()) if a.real_config else None)


if __name__=='__main__':main()
