# SPDX-License-Identifier: GPL-3.0-only
"""Small CPU fixtures for study v2. Never reads scientific main test objects."""
import argparse
import copy
import tempfile
import xml.etree.ElementTree as ET
from unittest.mock import patch
from pathlib import Path
import numpy as np
import torch
from scipy.stats import bootstrap
from dpnn_experiments import study_plan as plan
from dpnn_experiments import study_queue as queue
from dpnn_experiments.study_report import build
from dpnn_experiments.main_integrity import read, implementation
from dpnn_experiments.training_archive import file_reference, verify_reference
from dpnn_experiments.provenance import write_json
from dpnn_experiments.paired_statistics import bca_mean, paired_effect
from dpnn_experiments.evaluator import error_decomposition


def run(fixture,root):
    root=Path(root);root.mkdir(parents=True,exist_ok=False);checks=[]
    x=np.array([-.9,-.4,-.2,.1,.3,.7,1.1,2.8])
    ours=bca_mean(x)
    reference=bootstrap((x,),np.mean,method='BCa',n_resamples=10000,confidence_level=.95,random_state=np.random.default_rng(20261103),batch=128)
    np.testing.assert_allclose(ours['interval'],[reference.confidence_interval.low,reference.confidence_interval.high],rtol=1e-12,atol=1e-12)
    assert bca_mean([2,2,2])['status']=='degenerate_constant_effect'
    assert bca_mean([1])['interval'] is None
    assert not paired_effect({'a':1},{},['a'])['complete']
    assert paired_effect({'a':1},{},['a'])['bca']['interval'] is None
    checks+=['BCa agrees with SciPy at 10000 common resamples','constant, singleton and missing-pair cases explicit']
    class NonOrthogonal:
        def components(self,e):return .3*e,.7*e
    e=torch.tensor([[[[1.,2.],[3.,4.]]]],dtype=torch.float64)
    metrics,_=error_decomposition(NonOrthogonal(),e,torch.zeros_like(e),{'atol':1e-12,'rtol':1e-10})
    assert metrics['error_decomposition_passed'] and metrics['error_cross_term']>0
    np.testing.assert_allclose(sum(metrics[k] for k in ('error_null_mse','error_range_mse','error_cross_term')),7.5)
    checks.append('nonorthogonal numerical projector cross term and exact MSE identity')
    old=read(Path(fixture)/'plan.json');profile=read(plan.SOURCE/'configs/study/main_study_256.json')
    # Two interrupted raw-data generations must copy qualified complete batches,
    # not recompute them or change the resulting sample arrays.
    from dpnn_experiments.development_pilot import resolved, stage
    from dpnn_experiments.main_inputs import prepare_stage
    from dpnn_experiments.main_integrity import SealedArchive
    dc,_=resolved(old);dc['computation']['preparation_batch_size']=2
    preparation=root/'raw_recovery';result=preparation/'result.json';calls=[]
    def interrupted_stage(kind,config,folder,operator_root=None):
        config=copy.deepcopy(config);calls.append(1)
        if len(calls)<=2:config['stop_after_batches']=len(calls)
        return stage(kind,config,folder,operator_root)
    audit=lambda p:SealedArchive(p).member('dataset.json')
    for _ in range(2):
        try:prepare_stage('data',dc,preparation,result,interrupted_stage,audit,old['operator_root'])
        except InterruptedError:pass
        else:raise AssertionError('Planned raw data interruption missing')
    recovered=prepare_stage('data',dc,preparation,result,interrupted_stage,audit,old['operator_root'])
    original=stage('data',dc,root/'raw_reference',old['operator_root'])
    for a,b in zip(read(recovered/'dataset.json')['records'],read(original/'dataset.json')['records']):
        with np.load(a['file']['path']) as x,np.load(b['file']['path']) as y:
            for key in x.files:np.testing.assert_array_equal(x[key],y[key])
    assert sum(read(p)['reused'] for p in (recovered/'batches').glob('*.json'))==2
    checks.append('raw data resumes two committed batch prefixes bitwise; no regeneration of accepted batches')
    seeds=old['training_seeds']
    profile.update(kind='fixture',study_id=old['study_id'],base_plan=old,counts=old['counts'],epochs=2,
        data_seeds=old['data_seeds'],tv_max_iterations=old['starts']['tv']['max_iterations'],
        training_seed_sets=[seeds,{k:v+1 for k,v in seeds.items()}],exercise_stop_resume=True)
    study=plan.prepare(profile,root/'study')
    raw=next((Path(fixture)/'stages/data').glob('attempt_*/runs/*/dataset.json'))
    caches={s:next((Path(fixture)/('stages/start_'+s)).glob('attempt_*/runs/*/dataset.json')) for s in ('tikhonov','tv')}
    refs={'raw_dataset':file_reference(raw),'start_caches':{s:file_reference(p) for s,p in caches.items()}}
    plan.bind_inputs(study,refs)
    try:plan.freeze_tests(study)
    except (ValueError,FileNotFoundError):checks.append('premature test freeze rejected')
    else:raise AssertionError('Premature freeze accepted')
    bad=copy.deepcopy(profile);bad['beta_multipliers']=[0.,1.]
    try:plan.prepare(bad,root/'forbidden')
    except ValueError:checks.append('beta-zero study variation rejected')
    else:raise AssertionError('Forbidden budget variation accepted')
    with queue.locked(study/'controller.lock'):
        try:queue.run(study,'validation',cpu=True)
        except BlockingIOError:checks.append('duplicate controller lock rejected')
        else:raise AssertionError('Duplicate controller accepted')
    state=queue.start(study,'validation',cpu=True,foreground=True)
    assert state['status']=='interrupted',state
    assert len(state['failed'])==16,state
    for job in read(study/'validation_jobs.json'):
        if job['kind']=='train':
            cps=list((study/'jobs'/job['id']).glob('attempt_*/runs/*/latest_checkpoint.json'))
            assert len(cps)==1;verify_reference(read(cps[0]))
    checks.append('four CPU slots, all 16 training jobs safely stopped after optimizer step 1')
    state=queue.start(study,'validation',cpu=True,foreground=True)
    assert state['status']=='completed',state
    assert len(state['completed'])==50,state
    checks.append('all 16 trainings resumed; 2 BASE and 32 validation selections complete')
    validation=read(build(study,'validation'));assert validation['complete'],validation['issues']
    report_folder=Path(read(study/'validation_report.json')['path']).parent
    for plot in report_folder.glob('*.svg'):
        tree=ET.parse(plot)
        if 'learning' in plot.name or 'training_loss' in plot.name:
            assert tree.findall('.//{http://www.w3.org/2000/svg}circle'),plot
    resources=read(report_folder/'resource_logs.json')
    assert all(r['step_log'] for r in resources)
    for comparison in read(report_folder/'paired_effects.json'):
        assert comparison['bca']['n_objects']==old['counts']['validation']
    checks.append('nonempty training/validation curves, step resource references, and object-not-seed resampling counts')
    frozen=plan.freeze_tests(study);ref=file_reference(frozen);plan.freeze_tests(study);assert ref==file_reference(frozen)
    assert not (study/'jobs/test_BASE_tv').exists()
    checks.append('idempotent freeze binds all 34 test jobs without evaluating any test fixture')
    state=queue.start(study,'test',cpu=True,foreground=True);assert state['status']=='completed',state
    report=read(build(study,'test'));assert report['complete'],report['issues']
    assert report['expected_rows']==34*old['counts']['test'] and report['resume_exercised']
    assert report['gpu_count']==0
    checks.append('complete CPU-only artificial test evaluation, decomposition, aggregation and report')
    count_before=len(list((study/'jobs').glob('*/attempt_*')))
    queue.start(study,'test',cpu=True,foreground=True)
    assert len(list((study/'jobs').glob('*/attempt_*')))==count_before
    reused=sum('reused_for' in read(p) for p in (study/'jobs').glob('*/result.json'))
    assert reused>0
    checks.append('completed jobs skipped; identical selected weights reused with both logical bindings')
    for ref in [refs['raw_dataset'],*refs['start_caches'].values()]:verify_reference(ref)
    checks.append('shared source datasets unchanged')
    # A missing logical selection must remain visible even when its physical
    # archive exists; statistics may not silently treat available pairs as all.
    missing=study/'jobs/test_tikhonov_seed0_DP_best_mse/result.json';saved=missing.read_bytes();missing.unlink()
    incomplete=read(build(study,'test'))
    assert not incomplete['complete'] and any('test_tikhonov_seed0_DP_best_mse' in s for s in incomplete['issues'])
    missing.write_bytes(saved)
    checks.append('missing logical result is reported explicitly, inference not silently complete-case filtered')
    # Exact test contract tampering is rejected before any model call.
    frozen_index=read(study/'test_protocol_index.json');config_path=Path(frozen_index['evaluations'][0]['config']['path'])
    saved=config_path.read_bytes();config_path.write_bytes(saved+b' ')
    try:queue.config_for(study,frozen_index['evaluations'][0]['job'])
    except ValueError:checks.append('changed frozen evaluation rejected by hash')
    else:raise AssertionError('Changed test config accepted')
    config_path.write_bytes(saved)
    with patch.object(queue,'occupied',return_value={'GPU-foreign'}),patch.object(queue.os,'kill') as kill:
        queue.stop(study);kill.assert_not_called()
    checks.append('stop of inactive suite does not signal unrelated processes')
    # Restore final complete report after the deliberate missing-result check.
    build(study,'test')
    result={'passed':True,'checks':checks,'implementation':implementation(),'cpu_fixture_only':True,
        'gpu_rehearsal_executed':False,'main_started':False,'study':str(study),'report':file_reference(Path(read(study/'test_report.json')['path'])),
        'reused_evaluations':reused}
    write_json(root/'qualification.json',result);return root/'qualification.json'

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--fixture',type=Path,required=True);parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();print(run(args.fixture,args.root))
