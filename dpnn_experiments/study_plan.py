"""Versioned single-noise study preparation. No numerical work at prepare time."""
import copy
import os
from pathlib import Path
import numpy as np

from dpnn_experiments.development_pilot import resolved
from dpnn_experiments.evaluator import METRICS, protocol, validate
from dpnn_experiments.main_integrity import SealedArchive, implementation, read
from dpnn_experiments.provenance import write_json
from dpnn_experiments.training_archive import file_reference, verify_reference, canonical_hash

SOURCE=Path(__file__).resolve().parents[1]
METHODS=('RES','NSN','DP','DP_TWO')
SELECTIONS=('best_mse','best_ssim')
ANALYSIS={'schema':1,'error_decomposition':True,'retain_violations':True,
    'bootstrap':{'method':'BCa','resamples':10000,'seed':20261103,'confidence':.95},
    'comparisons':[['DP','BASE'],['DP','NSN'],['DP','RES'],['DP_TWO','DP']],
    'paired_metrics':['mse','ssim','psnr','relative_l2','data_change_over_noise','data_residual',
        'error_null_mse','error_range_mse','error_cross_term'],
    'seed_aggregation':'Average paired differences per object across seeds before resampling objects',
    'standard_deviations':['population_std','sample_std'],
    'missing_pairs':'No CI for incomplete or nonfinite pairs; retain explicit case ledger',
    'interpretation':'Object uncertainty for these models, not independent seeds or evidence of equivalence',
    'examples':{'fixed_count':8,'difficult_count':3,'difficult_rule':'Largest MSE per method; post-hoc descriptive only'},
    'image_scale':[0.,1.], 'metric_clipping':False,
    'error_space':'Bound numerical projector in pixel coordinates; not a continuous missing Fourier wedge'}


def expected_jobs(profile,phase):
    result=[]
    for start in ('tikhonov','tv'):
        result.append({'id':f'{phase}_BASE_{start}','kind':'evaluate','phase':phase,'start':start,
            'method':'BASE','seed_index':None,'selection':None,'dependencies':[]})
        for seed in range(len(profile['training_seed_sets'])):
            for method in METHODS:
                train=f'train_{start}_seed{seed}_{method}'
                if phase=='validation':
                    result.append({'id':train,'kind':'train','phase':phase,'start':start,'method':method,
                        'seed_index':seed,'selection':None,'dependencies':[]})
                for selection in SELECTIONS:
                    result.append({'id':f'{phase}_{start}_seed{seed}_{method}_{selection}','kind':'evaluate',
                        'phase':phase,'start':start,'method':method,'seed_index':seed,'selection':selection,
                        'dependencies':[train,f'{phase}_BASE_{start}']})
    # Long trainings first; all ready jobs still share a single dynamic queue.
    order={'DP_TWO':0,'DP':1,'NSN':2,'RES':3,'BASE':4}
    return sorted(result,key=lambda j:(j['kind']!='train',order[j['method']],j['id']))


def prepare(profile,root):
    root=Path(root).resolve();profile=copy.deepcopy(profile)
    if profile.get('schema_version')!=2 or profile.get('kind') not in ('main','rehearsal','fixture'):
        raise ValueError('Explicit version-2 study kind required')
    if profile.get('beta_multipliers')!=[1.0]:raise ValueError('Only beta=delta is authorized')
    seeds=profile['training_seed_sets']
    if len({canonical_hash(s) for s in seeds})!=len(seeds):raise ValueError('Duplicate training seeds')
    if len(seeds)!=(2 if profile['kind']=='main' else 1 if profile['kind']=='rehearsal' else len(seeds)):
        raise ValueError('Wrong seed count')
    base=copy.deepcopy(profile.get('base_plan')) if 'base_plan' in profile else read(SOURCE/profile['pilot_plan'])
    for key in ('external_inputs','selected_starts','continuation_of'):base.pop(key,None)
    base.update(study_id=profile['study_id'],counts=profile['counts'],epochs=profile['epochs'],
        data_seeds=profile['data_seeds'],training_seeds=seeds[0],methods=list(METHODS),
        evaluation_beta_multipliers=[1.0],cpu_threads=4)
    base['starts']['tv']['max_iterations']=profile['tv_max_iterations']
    data,training=resolved(base)
    data['purpose']='Independent main-study population; scientific test reserved until exact protocol freeze' if profile['kind']=='main' else 'Technical fixture population; not independent main-study evidence'
    if profile['kind']!='fixture':
        g=data['geometry']
        if (g['size'],g['angles'],g['detectors'],g['angle_min_deg'],g['angle_max_deg'])!=(256,120,363,-60.,60.):
            raise ValueError('Qualified 256-square geometry required')
    if profile['kind']=='main':
        g=data['geometry']
        if (profile['counts']!={'train':2500,'validation':500,'test':500} or profile['epochs']!=30
                or (g['size'],g['angles'],g['detectors'],g['angle_min_deg'],g['angle_max_deg'])!=(256,120,363,-60.,60.)):
            raise ValueError('Main scientific design changed')
    if profile['kind']!='fixture':
        if (training['training']['batch_size']!=2 or training['training']['gradient_clip_norm']!=1.
                or training['optimizer']['lr']!=1e-4 or training['scheduler']['step_size']!=10
                or training['scheduler']['gamma']!=.5 or base['relative_noise_norm']!=.05
                or base['starts']['tikhonov']['alpha']!=.01 or base['starts']['tv']['alpha']!=.001):
            raise ValueError('Qualified hyperparameters changed')
    if data['computation']['beta_noise_multiplier']!=1.:raise ValueError('Training beta must equal delta')
    if training['logging']['weights_every_step']:raise ValueError('Full per-step weights not authorized')
    if profile['kind']!='fixture' and (training['training']['mixed_precision'] or training['computation']['tf32']
            or training['computation']['dtype']!='float64' or training['training']['gradient_accumulation']!=1
            or training['training']['loss']!='mean_pixel_MSE' or training['selection']['patience']!=10
            or training['selection']['min_delta']!=0 or training['selection']['metric']!='mse'
            or profile['tv_max_iterations']!=200000):raise ValueError('Qualified precision/selection/solver contract changed')
    ids=copy.deepcopy(profile.get('object_ids',data['dataset']['split_ids']))
    flat=[i for group in ids.values() for i in group]
    if set(ids)!={'train','validation','test'} or len(flat)!=len(set(flat)) or any(not x for x in ids.values()):
        raise ValueError('Disjoint nonempty object sets required')
    if {s:len(v) for s,v in ids.items()}!=profile['counts']:raise ValueError('Count/ID mismatch')
    if profile['kind']=='rehearsal' and any('/test/' in i for i in flat):
        raise ValueError('Rehearsal must not use reserved scientific test objects')
    data['dataset']['split_ids']=ids
    root.mkdir(parents=True,exist_ok=False)
    analysis=copy.deepcopy(ANALYSIS)
    write_json(root/'profile.json',profile);write_json(root/'base_plan.json',base)
    write_json(root/'data_config.json',data);write_json(root/'training_template.json',training)
    write_json(root/'split_ids.json',ids);write_json(root/'analysis_plan.json',analysis)
    write_json(root/'protocol_design.json',{'selection_rules':list(SELECTIONS),'beta_multipliers':[1.0],
        'profile':file_reference(root/'profile.json'),'split_ids':file_reference(root/'split_ids.json'),
        'analysis_plan':file_reference(root/'analysis_plan.json'),
        'example_ids':{s:v[:analysis['examples']['fixed_count']] for s,v in ids.items() if s!='train'},
        'training_seeds':seeds,'tolerances':base['evaluation_tolerances'],
        'test_access':'Only explicit separate test command after freezing every selection'})
    for phase in ('validation','test'):write_json(root/f'{phase}_jobs.json',expected_jobs(profile,phase))
    files=['profile','base_plan','data_config','training_template','split_ids','analysis_plan',
        'protocol_design','validation_jobs','test_jobs']
    write_json(root/'definition.json',{'schema':2,'files':{n:file_reference(root/(n+'.json')) for n in files},
        'implementation':implementation(),'retention':'All artifacts local on this server; no external backup gate',
        'gpu_policy':'User authorizes four H200; one worker per GPU; respect foreign processes',
        'main_auto_start':False})
    write_json(root/'readiness.json',{'prepared':True,'inputs_bound':False,'started':False,
        'kind':profile['kind'],'external_backup_required':False,'gpu_general_rehearsal_executed':False})
    return root


def verify_definition(root):
    definition=read(root/'definition.json')
    for ref in definition['files'].values():verify_reference(ref)
    if definition['implementation']!=implementation():raise ValueError('Code differs from prepared version; prepare a new version')
    return definition


def bind_inputs(root,refs):
    """Verify exact data and immutable start arrays once; no solver or network."""
    verify_definition(root);profile=read(root/'profile.json');base=read(root/'base_plan.json')
    ids=read(root/'split_ids.json');dc=read(root/'data_config.json')
    raw_path=verify_reference(refs['raw_dataset']);raw_archive=SealedArchive(raw_path.parent)
    raw_archive.reference(refs['raw_dataset']);raw=read(raw_path)
    if raw['geometry']!=dc['geometry'] or raw['operator_sha256']!=base['operator_sha256']:
        raise ValueError('Input operator/geometry differs')
    if raw['configuration']['computation']['relative_noise_norm']!=base['relative_noise_norm']:
        raise ValueError('Input noise differs')
    if profile['kind']=='main' and (raw['split_ids']!=ids or raw['configuration']['randomness']!=dc['randomness']
            or raw['configuration']['study_id']!=profile['study_id']):
        raise ValueError('Main inputs must be the independent main population')
    records={r['id']:r for r in raw['records']};audited=[]
    raw_ids=[i for names in raw['split_ids'].values() for i in names]
    if len(records)!=len(raw['records']) or len(raw_ids)!=len(set(raw_ids)) or set(raw_ids)!=set(records):
        raise ValueError('Duplicate/missing raw IDs')
    for split,names in ids.items():
        expected_split='validation' if profile['kind']=='rehearsal' and split=='test' else split
        if not set(names)<=set(raw['split_ids'][expected_split]):raise ValueError('Input split mismatch')
    for start in ('tikhonov','tv'):
        path=verify_reference(refs['start_caches'][start]);archive=SealedArchive(path.parent)
        archive.reference(refs['start_caches'][start]);binding=read(path)
        if binding['parent_dataset']!=refs['raw_dataset'] or binding['start_reconstruction']!=base['starts'][start]:
            raise ValueError('Cache parent/solver differs')
        cached={r['id']:r for r in binding['records']}
        if binding['geometry']!=raw['geometry'] or binding['operator_sha256']!=raw['operator_sha256'] or binding['split_ids']!=raw['split_ids']:
            raise ValueError('Cache geometry/operator/split differs')
        if len(cached)!=len(binding['records']) or set(cached)!=set(records):raise ValueError('Duplicate/missing cache records')
        for name in [i for names in ids.values() for i in names]:
            original=records[name];record=cached[name]
            for ref in (original['file'],record['file'],record['start_solver']):verify_reference(ref)
            if not read(record['start_solver']['path']).get('passed'):raise ValueError('Unqualified cached start')
            with np.load(original['file']['path'],allow_pickle=False) as x,np.load(record['file']['path'],allow_pickle=False) as y:
                if any(not np.array_equal(x[k],y[k]) for k in ('target','measurement','analytic_clean','raster_clean','delta','beta')):
                    raise ValueError('Cache and raw sample differ')
                if not np.array_equal(x['beta'],x['delta']):raise ValueError('Cached training beta differs from delta')
                if y['start'].shape!=(1,dc['geometry']['size'],dc['geometry']['size']) or y['start'].dtype!=np.float64 or not np.isfinite(y['start']).all():raise ValueError('Invalid start')
        audited.append(file_reference(path.parent/'completion.json'))
        # Protect newly generated main inputs; historical pilot archives stay untouched.
        if root/'inputs' in path.parents:
            for file in path.parent.rglob('*'):
                if file.is_file():file.chmod(file.stat().st_mode & ~0o222)
    if root/'inputs' in raw_path.parents:
        for file in raw_path.parent.rglob('*'):
            if file.is_file():file.chmod(file.stat().st_mode & ~0o222)
    receipt={'inputs':refs,'definition':file_reference(root/'definition.json'),
        'completions':[file_reference(raw_path.parent/'completion.json'),*audited],
        'selected_counts':profile['counts'],'beta_equals_delta':True,'scientific_test_inference':False}
    if (root/'inputs.json').exists():
        if read(root/'inputs.json')!=receipt:raise ValueError('Inputs already bound differently')
    else:write_json(root/'inputs.json',receipt)
    return receipt


def training_config(root,job):
    tc=read(root/'training_template.json');ids=read(root/'split_ids.json');profile=read(root/'profile.json')
    tc['study_id']=profile['study_id'];tc['purpose']=f"Fixed study training train_{job['start']}_seed{job['seed_index']}_{job['method']}"
    tc['model']['method']=job['method'];tc['randomness'].update(profile['training_seed_sets'][job['seed_index']])
    tc['dataset']['reference']=read(root/'inputs.json')['inputs']['start_caches'][job['start']]
    tc['dataset']['used_ids']={s:ids[s] for s in ('train','validation')}
    if profile.get('epoch_extension'):
        ident=f"train_{job['start']}_seed{job['seed_index']}_{job['method']}"
        tc['execution']['continuation']=file_reference(root/'transitions'/(ident+'.json'))
    return tc


def job_result(root,job_id):
    result=read(root/'jobs'/job_id/'result.json')
    verify_reference(result['completion']);verify_reference(result['artifacts'])
    archive=SealedArchive(result['run'])
    if result['completion']!=file_reference(archive.root/'completion.json') or result['artifacts']!=file_reference(archive.root/'artifacts.json'):
        raise ValueError('Result points outside its sealed archive')
    sources={v['path']:v['sha256'] for v in read(archive.member('manifest.json'))['source_files']}
    if any(sources.get(k)!=v for k,v in read(root/'definition.json')['implementation'].items()):
        raise ValueError('Completed job source differs from prepared implementation')
    return result


def evaluation_config(root,job):
    profile=read(root/'profile.json');ids=read(root/'split_ids.json');base=read(root/'base_plan.json')
    inputs=read(root/'inputs.json')['inputs'];models=[]
    tc=training_config(root,{**job,'method':'RES','seed_index':0}) if job['method']=='BASE' else training_config(root,job)
    if job['method']!='BASE':
        train_id=f"train_{job['start']}_seed{job['seed_index']}_{job['method']}"
        result=job_result(root,train_id);archive=SealedArchive(result['run'])
        actual=copy.deepcopy(archive.configuration)
        for k in ('checkpoint_selection','dp_contract','retention_contract','model_evaluation'):actual.pop(k,None)
        actual['execution']['resume']=None;actual['execution']['stop_after_step']=None
        if actual!=tc:raise ValueError('Completed training differs from fixed configuration')
        best=read(archive.member(job['selection']+'.json'));archive.reference(best['weights'])
        models=[{'method':job['method'],'trained_start':job['start'],
            'training_manifest':file_reference(archive.member('manifest.json')),'weights':best['weights']}]
    phase=job['phase'];split='validation' if profile['kind']=='rehearsal' else phase
    config={'schema_version':1,'purpose':'Fixed study evaluation '+job['id'],'dataset':{'reference':inputs['raw_dataset']},
        'geometry':tc['geometry'],'randomness':tc['randomness'],'computation':tc['computation'],
        'model_config':tc,'models':models,'starts':[{**base['starts'][job['start']],'cache':inputs['start_caches'][job['start']]}],
        'beta_multipliers':[1.0],'split':split,'ids':ids[phase],
        'diagnostic_ids':read(root/'protocol_design.json')['example_ids'][phase],
        'allow_start_transfer':False,'metrics':{**METRICS,**tc['metrics']},'tolerances':base['evaluation_tolerances'],
        'frozen_protocol':None,'analysis_policy':read(root/'analysis_plan.json'),'include_baseline':job['method']=='BASE'}
    if job['method']!='BASE':
        baseline=job_result(root,f"{phase}_BASE_{job['start']}")
        config['baseline_reference']=file_reference(Path(baseline['run'])/'summary.json')
    return config


def evaluation_key(config):
    # Selection label/purpose is provenance, not a scientific evaluation choice.
    return canonical_hash(protocol(config))


def freeze_tests(root):
    verify_definition(root)
    validation=read(root/'validation_jobs.json')
    for job in validation:
        result=job_result(root,job['id']);archive=SealedArchive(result['run'])
        if job['kind']=='evaluate':
            expected=evaluation_config(root,job)
            if protocol(archive.configuration)!=protocol(expected):raise ValueError('Validation contract mismatch')
            if read(archive.member('summary.json'))['test_objects_read']!=0:raise ValueError('Premature test access')
    # Baseline references for test evaluations are not available until BASE is
    # evaluated. Freeze their job identities/definition now, not a result hash.
    rows=[];generation=root/'frozen_test'
    for job in read(root/'test_jobs.json'):
        if job['method']=='BASE':config=evaluation_config(root,job)
        else:
            # Construction needs the test BASE reference: bind instead to the
            # immutable baseline protocol. That reference remains fixed at run time.
            config=_test_model_config(root,job)
        folder=generation/job['id'];folder.mkdir(parents=True,exist_ok=True)
        frozen={'protocol':protocol(config),'selection_closed':True}
        for path,value in ((folder/'frozen.json',frozen),):
            if path.exists() and read(path)!=value:raise ValueError('Refusing to replace a frozen test contract')
            if not path.exists():write_json(path,value)
        config['frozen_protocol']=file_reference(folder/'frozen.json');validate(config)
        path=folder/'evaluation.json'
        if path.exists() and read(path)!=config:raise ValueError('Frozen evaluation changed')
        if not path.exists():write_json(path,config)
        rows.append({'job':job,'config':file_reference(path)})
    index={'definition':file_reference(root/'definition.json'),'inputs':file_reference(root/'inputs.json'),
        'evaluations':rows,'selection_closed':True,'test_quality_evaluated':False}
    path=root/'test_protocol_index.json'
    if path.exists() and read(path)!=index:raise ValueError('Published test index changed')
    if not path.exists():write_json(path,index)
    return path


def _test_model_config(root,job):
    # Reuse validation construction to bind the exact selected weights, then
    # replace only the predeclared evaluation population and baseline binding.
    config=evaluation_config(root,{**job,'phase':'validation'})
    config.update(ids=read(root/'split_ids.json')['test'],
        diagnostic_ids=read(root/'protocol_design.json')['example_ids']['test'],
        split='validation' if read(root/'profile.json')['kind']=='rehearsal' else 'test',
        purpose='Fixed study evaluation '+job['id'],
        baseline_reference={'job':f"test_BASE_{job['start']}",'definition':file_reference(root/'definition.json')})
    return config
