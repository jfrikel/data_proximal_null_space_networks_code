"""Archived recovery, exact replay and corruption rejection on small Radon data."""
import argparse,copy,json,shutil,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
from tests.start_qualification import run as fixtures
from dpnn_experiments.start_preparation import execute
from dpnn_experiments.training_archive import TrainingArchive, file_reference, audit_runs
from dpnn_experiments.provenance import write_json
SOURCE_ROOT=Path(__file__).resolve().parents[1]
def run(root):
    root=Path(root)
    c={'schema_version':1,'purpose':'Start recovery qualification with intentional failures','dataset':{'kind':'technical fixtures'},'geometry':{'size':8},'randomness':{'seed':3602},'computation':{'device':'cpu','dtype':'float64','cpu_threads':4}}
    with TrainingArchive(c,root/'runs',SOURCE_ROOT,root/'metadata') as a:
        fixture=fixtures(root/'fixture');s=json.loads((fixture/'summary.json').read_text())
        tv=root/'fixture/runs'/next(x['run'] for x in s['start_runs'] if x['kind']=='tv')
        config=json.loads((tv/'manifest.json').read_text())['configuration'];config['stop_after_objects']=2
        try:execute(config,root/'interrupt')
        except InterruptedError:pass
        else:raise AssertionError('Expected interruption')
        interrupted=next((root/'interrupt/runs').iterdir());assert not (interrupted/'dataset.json').exists()
        config.pop('stop_after_objects');config['start']['max_iterations']*=2
        config['reuse_completions']=[file_reference(interrupted/'completion.json')]
        recovered=execute(config,root/'recovered');rs=json.loads((recovered/'summary.json').read_text())
        assert rs['reused_objects']==2 and rs['computed_objects']==2
        old=json.loads((tv/'dataset.json').read_text());new=json.loads((recovered/'dataset.json').read_text());assert old['split_ids']==new['split_ids']
        for x,y in zip(old['records'],new['records']):
            with np.load(x['file']['path']) as u,np.load(y['file']['path']) as v:
                assert u.files==v.files
                for key in u.files:assert np.array_equal(u[key],v[key])
        checks={'interrupted_then_resumed_bitwise':True,'reused':2,'computed':2}
        sc=copy.deepcopy(config);sc.pop('reuse_completions')
        write_json(a.path/'signal_config.json',sc)
        with (a.path/'signal_process.log').open('w') as log:
            proc=subprocess.Popen([sys.executable,'-m','dpnn_experiments.start_preparation','--config',str(a.path/'signal_config.json'),'--root',str(root/'signal')],stdout=log,stderr=subprocess.STDOUT)
            deadline=time.monotonic()+60
            while proc.poll() is None and time.monotonic()<deadline:
                if list((root/'signal/runs').glob('*/objects/*.json')):
                    proc.send_signal(signal.SIGTERM);break
                time.sleep(.02)
            try:proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill();proc.wait();raise AssertionError('SIGTERM did not close archive')
        sr=next((root/'signal/runs').iterdir())
        assert json.loads((sr/'completion.json').read_text())['status']=='failed'
        resumed=copy.deepcopy(config);resumed['reuse_completions']=[file_reference(sr/'completion.json')]
        rr=execute(resumed,root/'signal_resumed')
        assert json.loads((rr/'summary.json').read_text())['reused_objects']>=1
        checks['real_sigterm_resume']=True

        for key,value in [('alpha',.002),('rtol',2e-5),('max_iterations',100)]:
            bad=copy.deepcopy(config);bad['start'][key]=value
            try:execute(bad,root/('reject_'+key))
            except ValueError:checks['rejected_'+key]=True
            else:raise AssertionError(key)
        one=copy.deepcopy(config);one.pop('reuse_completions');one['only_ids']=[old['records'][2]['id']]
        isolated=execute(one,root/'isolated');ib=json.loads((isolated/'dataset.json').read_text());assert len(ib['records'])==1
        joined=copy.deepcopy(config);joined['reuse_completions'].append(file_reference(isolated/'completion.json'))
        jr=execute(joined,root/'joined');js=json.loads((jr/'summary.json').read_text());assert js['reused_objects']==3 and js['computed_objects']==1
        checks['isolated_object_reused']=True
        damaged=root/'damaged';shutil.copytree(interrupted,damaged)
        with (damaged/'samples/00000.npz').open('ab') as f:f.write(b'corruption')
        bad=copy.deepcopy(config);bad['reuse_completions']=[file_reference(damaged/'completion.json')]
        try:execute(bad,root/'reject_corrupt')
        except ValueError:checks['rejected_corrupt_sample']=True
        else:raise AssertionError('corruption accepted')
        audits={d:audit_runs(root/d) for d in ['fixture','interrupt','recovered','signal','signal_resumed','isolated','joined','reject_alpha','reject_rtol','reject_max_iterations','reject_corrupt']}
        write_json(a.path/'summary.json',{'passed':True,'checks':checks,'archive_audits':audits,'corruption_fixture':str(damaged),'recovered':str(recovered)})
        print(json.dumps(checks),flush=True)
    return a.path
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);run(p.parse_args().root)
