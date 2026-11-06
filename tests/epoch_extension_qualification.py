# SPDX-License-Identifier: GPL-3.0-only
"""CPU-only epoch-extension, checkpoint selection, and four-worker replay tests."""
import argparse
import copy
from pathlib import Path
from unittest.mock import patch
import torch

from dpnn_experiments import epoch_extension as extension
from dpnn_experiments import study_plan as plan
from dpnn_experiments import study_queue as queue
from dpnn_experiments.main_integrity import read, implementation
from dpnn_experiments.provenance import write_json
from dpnn_experiments.training_archive import file_reference, verify_reference, canonical_hash
from tests.training_qualification import tensor_difference
from dpnn_experiments.study_report import training_segments, jsonlines
from dpnn_experiments import training


def state(run):
    return torch.load(verify_reference(read(run/'latest_checkpoint.json')),map_location='cpu',weights_only=True)


def steps(root,job):
    result={}
    for run,limit in training_segments(root,job):
        for row in jsonlines(run/'training_steps.jsonl'):
            if limit is None or row['step']<=limit[0]:result[row['step']]=row
    return [result[k] for k in sorted(result)]


def rejected(call):
    try:call()
    except (ValueError,FileNotFoundError):return True
    raise AssertionError('Invalid continuation accepted')


def run(parent,root):
    parent=Path(parent).resolve();root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False)
    receipt=root/'qualification.json';checks=[]
    write_json(receipt,{'passed':False,'implementation':implementation(),'phase':'in_progress'})
    # Only fixture approval is bypassed while qualifying its own queue. The real
    # production gate requires the final successful report and unchanged source.
    with patch.object(extension,'verify_qualification',return_value=None):
        study=extension.prepare(parent,root/'study',receipt,4)
        real_config=queue.config_for
        def stop_config(r,j):
            c=real_config(r,j)
            if j['kind']=='train':c['execution']['stop_after_step']=5
            return c
        with patch.object(queue,'config_for',side_effect=stop_config):
            stopped=queue.start(study,'validation',cpu=True,foreground=True)
        assert stopped['status']=='interrupted' and len(stopped['failed'])==16,stopped
        for job in read(study/'validation_jobs.json'):
            if job['kind']!='train':continue
            runs=list((study/'jobs'/job['id']).glob('attempt_*/runs/*/latest_checkpoint.json'))
            assert len(runs)==1
            s=state(runs[0].parent);assert s['progress']['step']==5 and s['progress']['epoch']==2
        checks.append('All 16 historical fixture endpoints extended from step 4; four CPU workers safely stop at step 5')
        resumed=queue.start(study,'validation',cpu=True,foreground=True)
        assert resumed['status']=='completed' and len(resumed['completed'])==50,resumed
        assert not (study/'test_protocol_index.json').exists() and not list((study/'jobs').glob('test_*'))
        checks.append('All 16 continuation jobs resumed to four epochs; 34 validation selections complete; no test work')
        count=len(list((study/'jobs').glob('*/attempt_*')))
        queue.start(study,'validation',cpu=True,foreground=True)
        assert count==len(list((study/'jobs').glob('*/attempt_*')))
        checks.append('Completed jobs are not repeated')
    # Full uninterrupted reference runs use the same original fixture population,
    # seeds and numerics; compare all optimizer steps across the original lineage.
    for method in plan.METHODS:
        job=next(j for j in read(study/'validation_jobs.json') if j['kind']=='train' and j['start']=='tikhonov' and j['seed_index']==0 and j['method']==method)
        c=plan.training_config(study,job);c['execution'].pop('continuation');c['execution']['resume']=None
        full=training.execute(c,root/('full_'+method));continued=Path(read(study/'jobs'/job['id']/'result.json')['run'])
        a=state(full);b=state(continued)
        for key in ('optimizer','scheduler','rng'):
            assert tensor_difference(a[key],b[key])==0,(method,key)
        wa=torch.load(verify_reference(a['weight_reference']),weights_only=True)
        wb=torch.load(verify_reference(b['weight_reference']),weights_only=True)
        assert tensor_difference(wa,wb)==0,method
        aa=jsonlines(full/'training_steps.jsonl');bb=steps(study,job)
        assert len(aa)==len(bb)==8
        for x,y in zip(aa,bb):
            for key in ('step','epoch','ids','learning_rates','loss','per_object_loss'):
                assert x[key]==y[key],(method,key)
        for key in ('best_mse','best_ssim'):
            for attr in ('epoch','step','mse','ssim'):assert a['progress'][key][attr]==b['progress'][key][attr]
        checks.append(method+': uninterrupted versus historical 2+2-epoch extension/stop/resume bitwise weights, Adam, scheduler, RNG, batches, losses and both best checkpoints')
    # Explicitly make later validation scores worse. Patience=1 would stop at
    # epoch two; disabled stopping must still reach epoch four and retain epoch 1.
    base=read(Path(read(study/'jobs/train_tikhonov_seed0_DP/result.json')['run'])/'manifest.json')['configuration']
    for k in ('checkpoint_selection','dp_contract','retention_contract','model_evaluation'):base.pop(k,None)
    base['execution'].pop('continuation');base['execution']['resume']=None
    base['selection'].update(patience=1,early_stopping_enabled=False);base['training']['epochs']=2
    original_evaluate=training.evaluate
    def worse(*args,**kwargs):
        result=original_evaluate(*args,**kwargs)
        if not kwargs.get('baseline',False):
            epoch=int(args[5])//2
            result['mse']['mean']=float(epoch);result['ssim']['mean']=1./epoch
        return result
    with patch.object(training,'evaluate',side_effect=worse):
        first=training.execute(base,root/'worse_parent')
        target=copy.deepcopy(base);target['training']['epochs']=4
        ref=extension.make_transition(first,target,root/'worse_transition.json')
        target['execution'].update(continuation=ref,resume=read(first/'latest_checkpoint.json'))
        second=training.execute(target,root/'worse_extension')
    s=state(second);assert s['progress']['epoch']==4 and s['progress']['bad_epochs']==3
    assert s['progress']['best_mse']['epoch']==s['progress']['best_ssim']['epoch']==0
    for key in ('best_mse','best_ssim'):
        saved=read(second/(key+'.json'));verify_reference(saved['weights'])
        assert Path(saved['weights']['path']).parent.parent==second
    checks.append('Patience=1 and deliberately worsening validation do not stop; original best MSE/SSIM weights retained and published')
    # Contract mutation tests do not train or read scientific test objects.
    parent_state=state(first);current=state(second)['binding'];contract=read(ref['path'])
    assert extension.validate_transition(target,parent_state,current,target['execution']['resume'])
    for section,key,value in [('optimizer','lr',.02),('randomness','model_seed',42),('training','batch_size',1),('scheduler','gamma',.1),('selection','early_stopping_enabled',True)]:
        bad=copy.deepcopy(target);bad[section][key]=value
        rejected(lambda:extension.validate_transition(bad,parent_state,current,bad['execution']['resume']))
    bad=copy.deepcopy(target);bad['execution'].pop('continuation')
    rejected(lambda:extension.validate_transition(bad,parent_state,current,bad['execution']['resume']))
    for key in ('torch','projection'):
        changed=copy.deepcopy(current);changed[key]='wrong'
        rejected(lambda:extension.validate_transition(target,parent_state,changed,target['execution']['resume']))
    badref=copy.deepcopy(target['execution']['resume']);badref['sha256']='0'*64
    rejected(lambda:extension.validate_transition(target,parent_state,current,badref))
    checks.append('Ordinary mismatched resume, changed LR/seed/batch/scheduler/stopping/environment/projector/checkpoint rejected')
    assert not any(read(p)['test_read_count'] for p in (root/'study/jobs').glob('train_*/attempt_*/runs/*/summary.json'))
    report={'passed':True,'implementation':implementation(),'all_four_methods_bitwise':True,
        'early_stopping_disabled_tested':True,'queue_resume_tested':True,'cpu_fixture_only':True,
        'scientific_test_read':False,'checks':checks,'fixture_parent':str(parent),'extension_study':str(study),
        'target_epochs':4,'production_target_epochs':60}
    write_json(receipt,report);return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();print(run(a.parent,a.root))
