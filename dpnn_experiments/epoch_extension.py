# SPDX-License-Identifier: GPL-3.0-only
"""Explicit, hash-bound continuation of a completed epoch budget.

This is not ordinary resume: only epoch budget and the early-stopping switch
may change. Original archives, model/data contracts and all optimizer/RNG state
are retained. The original scientific test results are not selection inputs.
"""
import argparse
import copy
from pathlib import Path
import torch

from dpnn_experiments.main_integrity import SealedArchive, implementation, read
from dpnn_experiments.provenance import write_json, utc_now
from dpnn_experiments.training_archive import canonical_hash, file_reference, verify_reference


def checked_parent(run):
    archive=SealedArchive(run)
    manifest=read(archive.member('manifest.json'))
    checkpoint=read(archive.member('latest_checkpoint.json'))
    archive.reference(checkpoint)
    state=torch.load(checkpoint['path'],map_location='cpu',weights_only=True,mmap=True)
    archive.reference(state['weight_reference'])
    from dpnn_experiments.training import scientific_configuration
    expected=canonical_hash(scientific_configuration(manifest['configuration']))
    sources=canonical_hash([x for x in manifest['source_files'] if x['path'].endswith('.py')])
    if state['binding']['configuration_sha256']!=expected or state['binding']['source_python_sha256']!=sources:
        raise ValueError('Parent checkpoint does not match its sealed manifest')
    if state['binding']!=read(archive.member('resume_binding.json')):
        raise ValueError('Parent resume binding differs')
    c=manifest['configuration'];p=state['progress']
    batches=(len(c['dataset']['used_ids']['train'])+c['training']['batch_size']-1)//c['training']['batch_size']
    if (p['phase']!='done' or p['epoch']!=c['training']['epochs'] or p['step']!=p['epoch']*batches
            or p['cursor']!=0 or p['order'] is not None):
        raise ValueError('Only a fully completed epoch-budget endpoint can be extended')
    for key in ('best_mse','best_ssim'):
        if p[key] is None:raise ValueError('Missing inherited selection')
        verify_reference(p[key]['weights'])
    return archive,manifest,checkpoint,state


def allowed_target(parent_config,target_config):
    from dpnn_experiments.training import finalized_config, scientific_configuration, validate_config
    validate_config(target_config)
    if target_config['selection'].get('early_stopping_enabled') is not False:
        raise ValueError('Extension must explicitly disable early stopping')
    if target_config['training']['epochs']<=parent_config['training']['epochs']:
        raise ValueError('Extension must increase total epochs')
    original=scientific_configuration(parent_config)
    expected=copy.deepcopy(original)
    expected['training']['epochs']=target_config['training']['epochs']
    expected['selection']['early_stopping_enabled']=False
    expected['checkpoint_selection']['early_stopping']='disabled_fixed_epoch_budget'
    actual=scientific_configuration(finalized_config(target_config))
    if actual!=expected:
        raise ValueError('Extension changes scientific parameters beyond epoch budget/early stopping')
    return canonical_hash(actual)


def make_transition(parent_run,target_config,path,parent_definition=None):
    archive,manifest,checkpoint,state=checked_parent(parent_run)
    target_hash=allowed_target(manifest['configuration'],target_config)
    result={'schema':1,'kind':'completed_epoch_budget_extension','created_utc':utc_now(),
        'parent_manifest':file_reference(archive.member('manifest.json')),
        'parent_completion':file_reference(archive.root/'completion.json'),
        'parent_artifacts':file_reference(archive.root/'artifacts.json'),
        'parent_checkpoint':checkpoint,'parent_binding':state['binding'],
        'parent_definition':parent_definition,'from_epochs':state['progress']['epoch'],
        'from_step':state['progress']['step'],'to_epochs':target_config['training']['epochs'],
        'target_configuration_sha256':target_hash,'target_implementation':implementation(),
        'early_stopping_enabled':False,'preserve_optimizer_scheduler_rng':True,
        'selection_scope':'all original and additional validation epochs; retain first on ties',
        'test_policy':'post-hoc extension, validation-only model selection; separate explicit test freeze required'}
    if parent_definition:
        d=read(verify_reference(parent_definition))
        sources={x['path']:x['sha256'] for x in manifest['source_files']}
        if any(sources.get(k)!=v for k,v in d['implementation'].items()):
            raise ValueError('Parent source differs from parent study definition')
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True);write_json(path,result)
    return file_reference(path)


def validate_transition(config,state,binding,resume_ref):
    ref=config['execution'].get('continuation')
    if not ref:raise ValueError('Resume configuration/code/environment/factor mismatch')
    t=read(verify_reference(ref))
    if (t['kind']!='completed_epoch_budget_extension' or t['parent_checkpoint']!=resume_ref
            or t['target_implementation']!=implementation()):
        raise ValueError('Unbound checkpoint or implementation transition')
    for k in ('parent_manifest','parent_completion','parent_artifacts'):
        verify_reference(t[k])
    archive=SealedArchive(Path(t['parent_manifest']['path']).parent)
    archive.reference(t['parent_manifest']);archive.reference(resume_ref)
    parent=archive.configuration
    if state['binding']!=t['parent_binding'] or allowed_target(parent,config)!=t['target_configuration_sha256']:
        raise ValueError('Parent or target transition binding mismatch')
    if binding['configuration_sha256']!=t['target_configuration_sha256']:
        raise ValueError('Target configuration differs from authorized transition')
    for k in ('torch','numpy','cuda','device_name','projection'):
        if state['binding'][k]!=binding[k]:raise ValueError('Extension changes environment/operator: '+k)
    p=state['progress']
    if (p['phase']!='done' or p['epoch']!=t['from_epochs'] or p['step']!=t['from_step']
            or p['cursor']!=0 or p['order'] is not None
            or t['to_epochs']!=config['training']['epochs'] or t['from_epochs']!=parent['training']['epochs']):
        raise ValueError('Invalid extension endpoint')
    return t


def verify_qualification(root):
    """Current CPU replay tests plus the already qualified numerical GPU parent."""
    root=Path(root);d=read(root/'definition.json');e=read(verify_reference(d['files']['epoch_extension']))
    q=read(verify_reference(e['qualification']))
    if not q.get('passed') or q.get('implementation')!=d['implementation']:
        raise ValueError('Current extension qualification required')
    if not q.get('all_four_methods_bitwise') or not q.get('early_stopping_disabled_tested') or not q.get('queue_resume_tested'):
        raise ValueError('Missing transition/stop-resume qualification')
    verify_reference(e['parent_definition']);verify_reference(e['parent_inputs'])
    if e['kind']=='main':
        old=read(verify_reference(e['parent_rehearsal_review']))
        report=read(verify_reference(old['report']))
        if not old['accepted'] or not report['complete'] or report['gpu_count']!=4 or not report['resume_exercised']:
            raise ValueError('Qualified four-H200 parent required')
    for ref in e['transitions'].values():
        t=read(verify_reference(ref))
        if t['target_implementation']!=d['implementation'] or t['to_epochs']!=read(root/'profile.json')['epochs']:
            raise ValueError('Transition/source/profile mismatch')


def prepare(parent,root,qualification,total_epochs=60):
    """Reuse the exact parent population/starts; prepare a new independent queue."""
    from dpnn_experiments import study_plan as plan
    from dpnn_experiments import study_queue as queue
    parent=Path(parent).resolve();root=Path(root).resolve()
    if root.exists():raise FileExistsError(root)
    if queue.controller_alive(parent):raise ValueError('Parent controller is still active')
    parent_d=read(parent/'definition.json');profile=read(parent/'profile.json')
    for ref in parent_d['files'].values():verify_reference(ref)
    queue.verify_inputs(parent)
    if profile['kind']=='main' and (profile['epochs']!=30 or total_epochs!=60
            or profile['counts']!={'train':2500,'validation':500,'test':500}
            or len(profile['training_seed_sets'])!=2 or profile['beta_multipliers']!=[1.]):
        raise ValueError('Only the explicitly authorized 16-run 30-to-60 main extension is supported')
    if profile['kind'] not in ('main','fixture'):raise ValueError('Unsupported extension population')
    # Parent configuration is reconstructed from its fixed study definition.
    parent_jobs=[j for j in read(parent/'validation_jobs.json') if j['kind']=='train']
    expected_count=4*2*len(profile['training_seed_sets'])
    if len(parent_jobs)!=expected_count:raise ValueError('Incomplete parent job design')
    parents={}
    for job in parent_jobs:
        result=plan.job_result(parent,job['id']);archive=SealedArchive(result['run'])
        actual=copy.deepcopy(archive.configuration)
        for key in ('checkpoint_selection','dp_contract','retention_contract','model_evaluation'):actual.pop(key,None)
        actual['execution']['resume']=None;actual['execution']['stop_after_step']=None
        if actual!=plan.training_config(parent,job):raise ValueError('Parent training differs from plan')
        parents[job['id']]=result['run']
    root.mkdir(parents=True)
    profile.update(study_id=profile['study_id']+'-epochs'+str(total_epochs),epochs=total_epochs,
                   exercise_stop_resume=False,epoch_extension=True,early_stopping_enabled=False)
    write_json(root/'profile.json',profile)
    for name in ('base_plan','data_config','training_template','split_ids','analysis_plan'):
        value=read(parent/(name+'.json'))
        if name=='base_plan':value['epochs']=total_epochs
        if name=='training_template':
            value['training']['epochs']=total_epochs;value['selection']['early_stopping_enabled']=False
        write_json(root/(name+'.json'),value)
    protocol=read(parent/'protocol_design.json')
    protocol.update(profile=file_reference(root/'profile.json'),split_ids=file_reference(root/'split_ids.json'),
                    analysis_plan=file_reference(root/'analysis_plan.json'),
                    continuation_policy='post-hoc 60-epoch fixed budget, no early stopping; checkpoints selected over all epochs',
                    test_access='Original test results already known; no automatic test execution or freeze')
    write_json(root/'protocol_design.json',protocol)
    for phase in ('validation','test'):write_json(root/(phase+'_jobs.json'),plan.expected_jobs(profile,phase))
    # Temporary receipt is only used to resolve configurations; no launch is
    # possible until the final immutable definition and qualification exist.
    inputs=read(parent/'inputs.json');write_json(root/'inputs.json',inputs)
    transitions={}
    for job in parent_jobs:
        # The continuation reference is excluded from the scientific hash and
        # is inserted by training_config only after this file exists.
        tc=read(root/'training_template.json');tc['study_id']=profile['study_id']
        tc['model']['method']=job['method'];tc['randomness'].update(profile['training_seed_sets'][job['seed_index']])
        tc['dataset']['reference']=inputs['inputs']['start_caches'][job['start']]
        tc['dataset']['used_ids']={s:read(root/'split_ids.json')[s] for s in ('train','validation')}
        transitions[job['id']]=make_transition(parents[job['id']],tc,root/'transitions'/(job['id']+'.json'),file_reference(parent/'definition.json'))
    extension={'schema':1,'kind':profile['kind'],'created_utc':utc_now(),'parent_root':str(parent),
        'parent_definition':file_reference(parent/'definition.json'),'parent_inputs':file_reference(parent/'inputs.json'),
        'parent_rehearsal_review':file_reference(parent/'rehearsal_review.json') if profile['kind']=='main' else None,
        'from_epochs':read(parent/'profile.json')['epochs'],'to_epochs':total_epochs,
        'early_stopping_enabled':False,'qualification':file_reference(qualification),'transitions':transitions,
        'user_authorization':'All 16 runs to 60 epochs without early stopping; prepare and start',
        'original_report_unchanged':True,'automatic_test_evaluation':False}
    write_json(root/'epoch_extension.json',extension)
    names=['profile','base_plan','data_config','training_template','split_ids','analysis_plan','protocol_design','validation_jobs','test_jobs','epoch_extension']
    write_json(root/'definition.json',{'schema':2,'files':{n:file_reference(root/(n+'.json')) for n in names},
        'implementation':implementation(),'retention':'All original and continuation archives retained on this server',
        'gpu_policy':'Four H200, one exclusive independent worker per GPU; respect foreign jobs','main_auto_start':False})
    inputs['definition']=file_reference(root/'definition.json');write_json(root/'inputs.json',inputs)
    write_json(root/'readiness.json',{'prepared':True,'inputs_bound':True,'kind':profile['kind'],
        'started':False,'epoch_extension':True,'external_backup_required':False})
    verify_qualification(root);plan.verify_definition(root);queue.verify_inputs(root)
    return root


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--qualification',type=Path,required=True);p.add_argument('--total-epochs',type=int,default=60)
    a=p.parse_args();print(prepare(a.parent,a.root,a.qualification,a.total_epochs))


if __name__=='__main__':main()
