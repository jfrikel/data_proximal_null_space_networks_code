# SPDX-License-Identifier: GPL-3.0-only
"""Exclusive device slots and a durable DAG queue. Never chains study phases."""
import argparse
import copy
import fcntl
import importlib.util
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
import traceback

from dpnn_experiments.main_integrity import read, SealedArchive
from dpnn_experiments.provenance import write_json, utc_now
from dpnn_experiments.training_archive import file_reference, verify_reference, append_json
from dpnn_experiments import study_plan as plan

SOURCE=Path(__file__).resolve().parents[1]


def launcher():
    spec=importlib.util.spec_from_file_location('pilot_launcher',SOURCE/'scripts/development_pilot.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def locked(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    f=path.open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);return f


def inventory(cpu=False):
    if cpu:return ['cpu-'+str(i) for i in range(4)]
    result=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,name','--format=csv,noheader,nounits'],check=True,capture_output=True,text=True)
    entries=[line.split(',') for line in result.stdout.splitlines()]
    if any('H200' not in row[2] for row in entries):raise ValueError('Qualified H200 device type required')
    ids=[row[1].strip() for row in entries]
    if len(ids)!=4:raise ValueError('Exactly four GPUs expected on the qualified server')
    return ids


def resources(root,profile,cpu=False):
    if cpu:return
    cores=len(os.sched_getaffinity(0))
    if cores<16:raise RuntimeError('Four workers require 16 available CPU cores; actual '+str(cores))
    memory={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')}
    free=shutil.disk_usage(root).free
    minimum=2*1024**4 if profile['kind']=='main' else 20*1024**3
    if free<minimum:raise RuntimeError('Insufficient local storage for retained study artifacts: '+str(free))
    if memory['MemAvailable']<160*1024**3:raise RuntimeError('Insufficient RAM for four projector workers')
    write_json(root/'resource_check.json',{'time':utc_now(),'available_cpu_cores':cores,'threads_per_worker':4,
        'available_ram_bytes':memory['MemAvailable'],'free_disk_bytes':free,'minimum_disk_bytes':minimum,
        'gpu_uuids':inventory(),'backup_required':False,'allocation_authority':'Four H200 workers specified by the study configuration'})


def occupied():
    result=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],check=True,capture_output=True,text=True)
    return {line.split(',')[0].strip() for line in result.stdout.splitlines() if line.strip()}


def verify_inputs(root):
    receipt=read(root/'inputs.json')
    if receipt['definition']!=file_reference(root/'definition.json'):raise ValueError('Input definition mismatch')
    for ref in receipt['completions']:verify_reference(ref)
    for ref in [receipt['inputs']['raw_dataset'],*receipt['inputs']['start_caches'].values()]:verify_reference(ref)


def main_review(root,profile):
    if profile.get('epoch_extension'):
        from dpnn_experiments.epoch_extension import verify_qualification
        verify_qualification(root)
        return
    if profile['kind']!='main':return
    approval=read(root/'rehearsal_review.json')
    if approval['definition']!=file_reference(root/'definition.json') or not approval['accepted']:
        raise ValueError('Review of completed rehearsal required before main work')
    report=read(verify_reference(approval['report']))
    if report.get('kind')!='rehearsal' or not report.get('complete') or report.get('phase')!='test' or report.get('gpu_count')!=4 or not report.get('resume_exercised'):
        raise ValueError('Completed four-GPU rehearsal with resume required')
    if report['implementation']!=read(root/'definition.json')['implementation']:raise ValueError('Rehearsal code mismatch')
    verify_reference(report['definition']);verify_reference(report['inputs'])
    for ref in report['input_references']:verify_reference(ref)


def snapshot(root):
    folder=root/'controller'
    if not folder.exists():launcher().setup(argparse.Namespace(root=folder,config=root/'base_plan.json'))
    launcher().verify_snapshot(folder)
    # The copied implementation must match this exact prepared version.
    expected=read(root/'definition.json')['implementation']
    actual={v['path']:v['sha256'] for v in read(folder/'launch_manifest.json')['source_files']}
    if any(actual.get(k)!=v for k,v in expected.items()):raise ValueError('Controller snapshot differs')
    return folder/'source'


def completed(root,job):
    path=root/'jobs'/job['id']/'result.json'
    if not path.exists():return False
    plan.job_result(root,job['id']);return True


def controller_alive(root):
    path=root/'process.json'
    return path.exists() and read(path).get('active',True) and launcher().alive(read(path))


def check_contract(root,job,run,expected):
    archive=SealedArchive(run);actual=copy.deepcopy(archive.configuration)
    if job['kind']=='train':
        for key in ('checkpoint_selection','dp_contract','retention_contract','model_evaluation'):actual.pop(key,None)
        actual['execution']['resume']=None;actual['execution']['stop_after_step']=None
        if actual!=expected:raise ValueError('Recovered training configuration mismatch')
    elif plan.protocol(actual)!=plan.protocol(expected):raise ValueError('Recovered evaluation protocol mismatch')
    sources={v['path']:v['sha256'] for v in read(archive.member('manifest.json'))['source_files']}
    if any(sources.get(k)!=v for k,v in read(root/'definition.json')['implementation'].items()):raise ValueError('Recovered source mismatch')


def config_for(root,job):
    if job['kind']=='train':return plan.training_config(root,job)
    if job['phase']=='validation':return plan.evaluation_config(root,job)
    index=read(root/'test_protocol_index.json')
    if index['definition']!=file_reference(root/'definition.json') or index['inputs']!=file_reference(root/'inputs.json'):
        raise ValueError('Frozen test inputs changed')
    records=[v for v in index['evaluations'] if v['job']==job]
    if len(records)!=1:raise ValueError('Missing/duplicate frozen test job')
    config=read(verify_reference(records[0]['config']))
    frozen=read(verify_reference(config['frozen_protocol']))
    if frozen['protocol']!=plan.protocol(config):raise ValueError('Frozen test contract changed')
    return config


def result_for(run):
    archive=SealedArchive(run)
    return {'status':'completed','run':str(run),'completion':file_reference(archive.root/'completion.json'),
        'artifacts':file_reference(archive.root/'artifacts.json')}


def execute_attempt(root,job,attempt):
    """Runs inside the device-isolated frozen source subprocess."""
    from dpnn_experiments.development_pilot import stage
    spec=read(attempt/'dispatch.json');config=read(verify_reference(spec['configuration']))
    run=stage(job['kind'],config,attempt)
    write_json(attempt/'outcome.json',{'run':str(run),'status':read(run/'completion.json')['status']})


def reap_orphans(root):
    l=launcher()
    for dispatch in (root/'jobs').glob('*/attempt_*/process.json'):
        record=read(dispatch)
        if l.alive(record):raise RuntimeError('Previous owned worker is still alive; stop it before resume: '+str(dispatch))


def run(root,phase,cpu=False):
    root=Path(root).resolve();guard=locked(root/'controller.lock')
    plan.verify_definition(root);profile=read(root/'profile.json')
    if cpu and profile['kind']!='fixture':raise ValueError('CPU slots are restricted to technical fixtures')
    main_review(root,profile)
    resources(root,profile,cpu);verify_inputs(root);source=snapshot(root);reap_orphans(root)
    if phase=='test':read(root/'test_protocol_index.json')
    jobs=read(root/f'{phase}_jobs.json');devices=inventory(cpu);running={};failed=set();blocked=set();configs={}
    def get_config(job):
        if job['id'] not in configs:configs[job['id']]=config_for(root,job)
        return configs[job['id']]
    stop={'requested':False}
    def stopping(*_):stop['requested']=True
    old={s:signal.signal(s,stopping) for s in (signal.SIGTERM,signal.SIGINT)}
    l=launcher();write_json(root/'process.json',{'pid':os.getpid(),'identity':l.identity(os.getpid()),'phase':phase,'active':True})
    (root/'STOP').unlink(missing_ok=True)
    def state(status):
        write_json(root/'status.json',{'status':status,'phase':phase,'time':utc_now(),
            'completed':[j['id'] for j in jobs if (root/'jobs'/j['id']/'result.json').exists()],
            'failed':sorted(failed),'blocked':sorted(blocked),'running':{d:r['job']['id'] for d,r in running.items()},
            'devices':devices,'expected_jobs':len(jobs),'main_auto_start':False})
    try:
        while True:
            if (root/'STOP').exists():stop['requested']=True
            for device,item in list(running.items()):
                process=item['process']
                if stop['requested'] and not item.get('signaled'):
                    process.send_signal(signal.SIGTERM);item['signaled']=True
                if process.poll() is None:continue
                item['log'].close();item['lock'].close();attempt=item['attempt'];job=item['job']
                if (attempt/'outcome.json').exists() and read(attempt/'outcome.json')['status']=='completed':
                    check_contract(root,job,Path(read(attempt/'outcome.json')['run']),get_config(job))
                    result=result_for(Path(read(attempt/'outcome.json')['run']))
                    result['assignment']=file_reference(attempt/'dispatch.json')
                    write_json(root/'jobs'/job['id']/'result.json',result)
                else:failed.add(job['id'])
                del running[device]
            active={item['job']['id'] for item in running.values()}
            pending=[j for j in jobs if not completed(root,j) and j['id'] not in failed|active|blocked]
            for job in pending:
                if any(dep in failed|blocked for dep in job['dependencies']):blocked.add(job['id'])
            pending=[j for j in pending if j['id'] not in blocked]
            if not stop['requested']:
                busy=set() if cpu else occupied()
                for device in devices:
                    if device in running or device in busy:continue
                    ready=[j for j in pending if all((root/'jobs'/dep/'result.json').exists() for dep in j['dependencies'])]
                    if not ready:break
                    # A duplicate selection waiting for an in-flight evaluation
                    # must not block unrelated ready work on the other devices.
                    candidate=None
                    for ready_job in ready:
                        ready_config=get_config(ready_job)
                        ready_key=plan.evaluation_key(ready_config) if ready_job['kind']=='evaluate' else None
                        if ready_key and any(v.get('key')==ready_key for v in running.values()):continue
                        candidate=(ready_job,ready_config);break
                    if candidate is None:continue
                    job,config=candidate;folder=root/'jobs'/job['id'];folder.mkdir(parents=True,exist_ok=True)
                    # Recover a committed stage after a controller crash before publication.
                    recovered=False
                    for marker in sorted(folder.glob('attempt_*/runs/*/completion.json'),reverse=True):
                        if read(marker)['status']=='completed':
                            check_contract(root,job,marker.parent,config)
                            write_json(folder/'result.json',result_for(marker.parent));recovered=True;break
                    if recovered:pending.remove(job);continue
                    key=plan.evaluation_key(config) if job['kind']=='evaluate' else None
                    if key:
                        reuse=None
                        for other in jobs:
                            if other['kind']=='evaluate' and completed(root,other):
                                candidate=plan.job_result(root,other['id'])
                                if plan.evaluation_key(SealedArchive(candidate['run']).configuration)==key:reuse=candidate;break
                        if reuse:
                            write_json(folder/'result.json',{**reuse,'reused_for':job,'evaluation_key':key});pending.remove(job);continue
                        if any(v.get('key')==key for v in running.values()):continue
                    try:device_lock=locked(Path('/tmp')/('dpnn-device-'+device+'.lock'))
                    except BlockingIOError:continue
                    if not cpu and device in occupied():device_lock.close();continue
                    attempts=sorted(folder.glob('attempt_*'));attempt=folder/f'attempt_{len(attempts)+1:04d}';attempt.mkdir()
                    actual=copy.deepcopy(config)
                    if job['kind']=='train':
                        cps=sorted(folder.glob('attempt_*/runs/*/latest_checkpoint.json'),key=lambda p:p.stat().st_mtime_ns)
                        if cps:actual['execution']['resume']=read(cps[-1]);verify_reference(actual['execution']['resume'])
                        elif actual['execution'].get('continuation'):
                            transition=read(verify_reference(actual['execution']['continuation']))
                            actual['execution']['resume']=transition['parent_checkpoint']
                            verify_reference(actual['execution']['resume'])
                        elif profile.get('exercise_stop_resume'):actual['execution']['stop_after_step']=1
                    write_json(attempt/'config.json',actual)
                    dispatch={'job':job,'gpu_uuid':device,'definition':file_reference(root/'definition.json'),
                        'inputs':file_reference(root/'inputs.json'),'configuration':file_reference(attempt/'config.json'),
                        'started':utc_now(),'cpu_threads':4,'source':str(source)}
                    write_json(attempt/'dispatch.json',dispatch)
                    env=os.environ.copy();env.update(PYTHONPATH=str(source),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4')
                    if not cpu:env['CUDA_VISIBLE_DEVICES']=device
                    log=(attempt/'process.log').open('a')
                    process=subprocess.Popen([sys.executable,'-u',str(source/'scripts/main_study.py'),'_attempt',str(root),'--attempt',str(attempt)],
                        cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,pass_fds=(device_lock.fileno(),))
                    write_json(attempt/'process.json',{'pid':process.pid,'identity':l.identity(process.pid)})
                    running[device]={'process':process,'job':job,'attempt':attempt,'lock':device_lock,'log':log,'key':key}
                    pending.remove(job)
            state('stopping' if stop['requested'] else 'running')
            if not running:
                if stop['requested'] or not pending:break
                if all(any(not (root/'jobs'/dep/'result.json').exists() for dep in j['dependencies']) for j in pending):
                    raise ValueError('Unresolved dependencies; complete validation before test')
                # Foreign occupancy is a concrete failure, not permission to displace it.
                raise RuntimeError('No available authorized GPU slots; inspect status and foreign processes, then resume')
            time.sleep(.2 if cpu else 2)
        state('interrupted' if stop['requested'] or failed else 'completed' if not blocked else 'blocked')
    except BaseException:
        for item in running.values():
            if item['process'].poll() is None:item['process'].send_signal(signal.SIGTERM)
        state('failed');raise
    finally:
        for s,h in old.items():signal.signal(s,h)
        write_json(root/'process.json',{'pid':os.getpid(),'identity':l.identity(os.getpid()),'phase':phase,'active':False})
        guard.close()
    return read(root/'status.json')


def start(root,phase,cpu=False,foreground=False):
    root=Path(root).resolve();plan.verify_definition(root)
    profile=read(root/'profile.json')
    if cpu and profile['kind']!='fixture':raise ValueError('CPU slots are restricted to technical fixtures')
    main_review(root,profile);verify_inputs(root);resources(root,profile,cpu)
    if not cpu and not set(inventory())-occupied():raise RuntimeError('All four GPUs are occupied by foreign jobs')
    if phase=='test':read(root/'test_protocol_index.json')
    with locked(root/'launch.lock'):
        with locked(root/'controller.lock'):reap_orphans(root)
        if controller_alive(root):raise RuntimeError('Controller already running')
        source=snapshot(root)
        if foreground:return run(root,phase,cpu)
        log=(root/'controller.log').open('a')
        args=[sys.executable,'-u',str(source/'scripts/main_study.py'),'_run',str(root),'--phase',phase]
        if cpu:args.append('--cpu-fixture')
        p=subprocess.Popen(args,cwd=source,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write_json(root/'process.json',{'pid':p.pid,'identity':launcher().identity(p.pid),'phase':phase});log.close()
        return {'pid':p.pid,'phase':phase,'log':str(root/'controller.log')}


def stop(root):
    write_json(root/'STOP',{'requested':utc_now()})
    # Only identities recorded by this study are signaled; never GPU strangers.
    for path in [root/'process.json',*root.glob('jobs/*/attempt_*/process.json')]:
        if path.exists():
            record=read(path)
            if record.get('active',True) and launcher().alive(record):os.kill(record['pid'],signal.SIGTERM)
    return {'stop_requested':True,'checkpoint':'At next safe training boundary; use status before resume'}


def prepare_inputs(root):
    from dpnn_experiments.development_pilot import stage
    from dpnn_experiments.main_inputs import prepare_stage
    from dpnn_experiments.pilot_review import audit
    plan.verify_definition(root)
    profile=read(root/'profile.json');cpu=profile['kind']=='fixture'
    if profile['kind']=='rehearsal':raise ValueError('Rehearsal consumes the fixed existing development caches; no generation')
    main_review(root,profile)
    resources(root,profile,cpu)
    devices=inventory(cpu);free=[d for d in devices if cpu or d not in occupied()]
    if not free:raise RuntimeError('No free GPU for input preparation; foreign jobs are respected')
    with locked(root/'controller.lock'),locked(root/'inputs.lock'),locked(Path('/tmp')/('dpnn-device-'+free[0]+'.lock')):
        if not cpu:
            if free[0] in occupied():raise RuntimeError('GPU acquired by a foreign process')
            os.environ['CUDA_VISIBLE_DEVICES']=free[0]
        for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='4'
        write_json(root/'process.json',{'pid':os.getpid(),'identity':launcher().identity(os.getpid()),'phase':'inputs','active':True})
        write_json(root/'input_assignment.json',{'gpu_uuid':free[0],'definition':file_reference(root/'definition.json'),'single_writer':True})
        def requested(*_):raise InterruptedError('Input preparation stopped; qualified prefixes retained')
        previous={s:signal.signal(s,requested) for s in (signal.SIGTERM,signal.SIGINT)}
        write_json(root/'status.json',{'status':'running','phase':'inputs','gpu_uuid':free[0]})
        try:
            result=_prepare_inputs_locked(root,stage,prepare_stage,audit)
            write_json(root/'status.json',{'status':'completed','phase':'inputs','gpu_uuid':free[0]})
            return result
        except BaseException as error:
            write_json(root/'status.json',{'status':'interrupted' if isinstance(error,InterruptedError) else 'failed','phase':'inputs','reason':str(error)})
            raise
        finally:
            for s,h in previous.items():signal.signal(s,h)
            write_json(root/'process.json',{'pid':os.getpid(),'identity':launcher().identity(os.getpid()),'phase':'inputs','active':False})


def _prepare_inputs_locked(root,stage,prepare_stage,audit):
        base=read(root/'base_plan.json');dc=read(root/'data_config.json');tc=read(root/'training_template.json')
        # One writer, stages sequential. Resumption reuses sealed qualified prefixes.
        raw=prepare_stage('data',dc,root/'inputs/data',root/'inputs/data/result.json',stage,lambda p:audit(p,True),base['operator_root'])
        refs={'raw_dataset':file_reference(raw/'dataset.json'),'start_caches':{}}
        for name in ('tikhonov','tv'):
            config={'schema_version':1,'purpose':'Fixed shared study cache','geometry':dc['geometry'],
                'randomness':dc['randomness'],'computation':tc['computation'],'operator_root':base['operator_root'],
                'dataset':{'reference':refs['raw_dataset']},'start':base['starts'][name]}
            cached=prepare_stage('start',config,root/f'inputs/{name}',root/f'inputs/{name}/result.json',stage,lambda p:audit(p,True))
            refs['start_caches'][name]=file_reference(cached/'dataset.json')
        return plan.bind_inputs(root,refs)
