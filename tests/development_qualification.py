# SPDX-License-Identifier: GPL-3.0-only
"""Archived CPU qualification of detached development orchestration and inherited best weights."""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from dataclasses import asdict
from dpnn_experiments.operators import Geometry, build_radon
from dpnn_experiments.provenance import RunArchive, write_json
from dpnn_experiments.training_archive import TrainingArchive, file_reference, verify_reference
from dpnn_experiments.gpu_benchmark import load_operator, reference_metadata
from dpnn_experiments.projections import MatrixOperator
from dpnn_experiments.training import execute
from tests.training_qualification import compare_runs
from dpnn_experiments.evaluator import execute as evaluate, METRICS

SOURCE=Path(__file__).resolve().parents[1]


def run(root):
    root=Path(root).resolve()
    cfg={'schema_version':1,'purpose':'CPU launcher pipeline, resume, inherited selection and guards; no scientific test',
         'geometry':{'size':8},'dataset':{'kind':'technical independent fixtures'},'randomness':{'seed':521},
         'computation':{'device':'cpu','dtype':'float64'}}
    with TrainingArchive(cfg,root/'runs',SOURCE,root/'metadata') as a:
        g=Geometry(size=8,angles=12,detectors=13,angle_min_deg=-60.,angle_max_deg=60.,half_width=2**.5)
        oc={**cfg,'geometry':asdict(g),'execution_action':'prepare'}
        oc['computation']={**cfg['computation'],'operator':'pixel_line_intersections_v1'}
        with RunArchive(oc,root/'operators',SOURCE,root/'operator_metadata') as oa:
            build_radon(g).save(oa.path/'operator.npz')
        op=MatrixOperator.whitened_radon(load_operator(oa.path,reference_metadata(oa.path)),'cpu')
        matrix=op.matrix.to_dense().numpy();_,s,v=np.linalg.svd(matrix,full_matrices=False)
        projection=a.path/'projection.npz'
        np.savez_compressed(projection,matrix=matrix,basis=v[s>1e-12*s[0]].T,image_weight=(2/8)**2)
        plan=json.loads((SOURCE/'configs/study/development_pilot_256.json').read_text())
        dc=json.loads((SOURCE/'configs/tests/pilot_data_256_cpu.json').read_text());dc['geometry']=asdict(g)
        tc=json.loads((SOURCE/'configs/tests/scalable_training_qualification.json').read_text());tc['model']['dropout']=0.
        plan.update(study_id='technical-launcher-fixture',data_configuration=dc,training_configuration=tc,
                    counts={'train':4,'validation':2,'test':1},device='cpu',epochs=3,patience=5,
                    operator_root=str(oa.path),operator_sha256=file_reference(oa.path/'operator.npz')['sha256'],
                    projection={'kind':'fixture','reference':file_reference(projection)},minimum_free_gib=0,
                    stage_free_reserve_gib=0,diagnostic_objects=1,learning_rate=0.)
        pc=a.path/'cpu_plan.json';write_json(pc,plan)
        study=a.path/'background_study';script=SOURCE/'scripts/development_pilot.py'
        commands=[]
        def command(*args):
            cmd=[sys.executable,str(script),*map(str,args)]
            out=subprocess.run(cmd,cwd=SOURCE,text=True,capture_output=True)
            commands.append({'command':cmd,'returncode':out.returncode,'stdout':out.stdout,'stderr':out.stderr})
            write_json(a.path/'commands.json',commands)
            return out
        start=command('start','--config',pc,'--root',study);assert start.returncode==0,start.stderr
        duplicate=command('resume','--root',study);assert duplicate.returncode!=0
        # Request stop while the first process is running; worker honours it between stages too.
        stopped=command('stop','--root',study);assert stopped.returncode==0
        def wait_status(terminal):
            deadline=time.monotonic()+240
            while time.monotonic()<deadline:
                if (study/'status.json').exists():
                    state=json.loads((study/'status.json').read_text())
                    if state['status'] in terminal:return state
                time.sleep(.1)
            raise TimeoutError('Worker did not finish')
        state=wait_status({'interrupted','failed','completed'});assert state['status']=='interrupted',state
        # Wait for lock release rather than racing the final status write.
        time.sleep(.5)
        resumed=command('resume','--root',study);assert resumed.returncode==0,resumed.stderr
        state=wait_status({'completed','failed'});assert state['status']=='completed',state
        summary=json.loads((study/'summary.json').read_text());assert summary['training_jobs']==8 and summary['evaluation_jobs']==16
        assert not summary['scientific_test_evaluated']
        # Inspect every evaluator, including both selection rules and both starts.
        for record in summary['evaluations']:
            es=json.loads(verify_reference(record['summary']).read_text())
            assert es['test_objects_read']==0 and es['split']=='validation'
        # Stop an actual training stage after its first best checkpoint, then let
        # the orchestrator restore it and finish both starts and both selections.
        train_plan=copy.deepcopy(plan)
        train_plan.update(study_id='technical-launcher-training-stop',methods=['DP_TWO'],
                          counts={'train':16,'validation':2,'test':1},epochs=20,patience=25)
        train_pc=a.path/'training_stop_plan.json';write_json(train_pc,train_plan)
        first_study=study
        study=a.path/'training_stop_study'
        launched=command('start','--config',train_pc,'--root',study);assert launched.returncode==0,launched.stderr
        deadline=time.monotonic()+120
        while time.monotonic()<deadline:
            best_files=list((study/'stages/train_tikhonov_DP_TWO').glob('attempt_*/runs/*/best_mse.json'))
            if best_files:break
            if (study/'status.json').exists() and json.loads((study/'status.json').read_text())['status']=='failed':
                raise AssertionError((study/'status.json').read_text())
            time.sleep(.02)
        else:raise TimeoutError('No training best checkpoint')
        command('stop','--root',study)
        state=wait_status({'interrupted','failed','completed'});assert state['status']=='interrupted',state
        time.sleep(.5)
        again=command('resume','--root',study);assert again.returncode==0,again.stderr
        state=wait_status({'completed','failed'});assert state['status']=='completed',state
        manifests=list((study/'stages/train_tikhonov_DP_TWO').glob('attempt_*/runs/*/manifest.json'))
        assert any(json.loads(p.read_text())['configuration']['execution']['resume'] for p in manifests)
        training_stop_study=str(study)
        study=first_study
        # An inherited epoch-one best must be publishable after a later interruption.
        original=Path(summary['evaluations'][0]['training_run'])
        training=json.loads((original/'manifest.json').read_text())['configuration']
        for key in ('checkpoint_selection','dp_contract','retention_contract','model_evaluation'):
            training.pop(key,None)
        part=copy.deepcopy(training);part['execution']['stop_after_step']=3
        first=execute(part,root);assert json.loads((first/'completion.json').read_text())['status']=='interrupted'
        rc=copy.deepcopy(training);rc['execution']['resume']=json.loads((first/'latest_checkpoint.json').read_text())
        final=execute(rc,root)
        comparison=compare_runs(original,first,final,0.)
        best=json.loads((final/'best_mse.json').read_text());assert 'source_weights' in best
        assert verify_reference(best['weights']).parent.parent==final
        assert best['weights']['sha256']==best['source_weights']['sha256']
        ec=json.loads((Path(summary['evaluations'][0]['evaluation_run'])/'manifest.json').read_text())['configuration']
        ec['models'][0].update(training_manifest=file_reference(final/'manifest.json'),weights=best['weights'])
        ev=evaluate(ec,root)
        assert json.loads((ev/'summary.json').read_text())['test_objects_read']==0
        # Resume must reject a modified source snapshot before launching any process.
        snap=study/'source/dpnn_experiments/development_pilot.py'
        content=snap.read_text();snap.write_text(content+'\n# deliberate corruption guard fixture\n')
        rejected=command('resume','--root',study);assert rejected.returncode!=0 and 'Artifact changed' in rejected.stderr
        snap.write_text(content)
        write_json(a.path/'summary.json',{'passed':True,'study':str(study),'stop_resume':True,
            'duplicate_launch_rejected':True,'snapshot_tampering_rejected':True,'training_jobs':8,'evaluation_jobs':16,
            'training_stage_stop_resume':training_stop_study,'inherited_best_evaluation_run':ev.name,'resume_comparison':comparison,'scientific_test_used':False})
    return a.path

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);a=p.parse_args();print(run(a.root))
