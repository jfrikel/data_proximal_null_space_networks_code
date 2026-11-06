# SPDX-License-Identifier: GPL-3.0-only
"""Small Radon end-to-end qualification of immutable TV/Tikhonov start caches."""
import argparse,copy,json
from dataclasses import asdict
from pathlib import Path
import numpy as np
from dpnn_experiments.operators import Geometry, build_radon
from dpnn_experiments.provenance import RunArchive, write_json
from dpnn_experiments.training_archive import TrainingArchive, file_reference
from dpnn_experiments.pilot_data import prepare
from dpnn_experiments.start_preparation import execute
from dpnn_experiments.training import CachedData
SOURCE_ROOT=Path(__file__).resolve().parents[1]


def run(root):
    root=Path(root);g=Geometry(size=8,angles=12,detectors=13,angle_min_deg=-60.,angle_max_deg=60.,half_width=2**.5)
    c={'schema_version':1,'purpose':'Small Radon start cache qualification, no scientific test evaluation','dataset':{'kind':'four independent technical phantoms'},'geometry':asdict(g),
       'randomness':{'seed':3602},'computation':{'device':'cpu','dtype':'float64','cpu_threads':4}}
    with TrainingArchive(c,root/'runs',SOURCE_ROOT,root/'metadata') as a:
        oc=copy.deepcopy(c);oc['execution_action']='prepare';oc['computation']['operator']='pixel_line_intersections_v1'
        with RunArchive(oc,root/'operators',SOURCE_ROOT,root/'operator_metadata') as oa:build_radon(g).save(oa.path/'operator.npz')
        pc=json.loads((SOURCE_ROOT/'configs/tests/pilot_data_256_cpu.json').read_text());pc['geometry']=asdict(g)
        pc['dataset']['split_ids']={'train':['startfixture/train/0','startfixture/train/1'],'validation':['startfixture/validation/0'],'test':['startfixture/test/0']}
        pc['operator_sha256']=file_reference(oa.path/'operator.npz')['sha256'];pc['computation']['tikhonov'].update(rtol=1e-9,atol=1e-11,max_iterations=500)
        data=prepare(pc,root,oa.path);binding=json.loads((data/'dataset.json').read_text());out=[]
        for settings in ({'kind':'tikhonov',**pc['computation']['tikhonov']},
                         {'kind':'tv','alpha':.001,'rtol':1e-5,'atol':1e-9,'max_iterations':20000,'check_every':50,'step_safety':.99}):
            sc={**c,'dataset':{'reference':file_reference(data/'dataset.json')},'operator_root':str(oa.path),'start':settings}
            result=execute(sc,root);nb=json.loads((result/'dataset.json').read_text())
            assert nb['split_ids']==binding['split_ids']
            errors=[]
            for old,new in zip(binding['records'],nb['records']):
                with np.load(old['file']['path']) as x,np.load(new['file']['path']) as y:
                    for k in x.files:
                        if k!='start':assert np.array_equal(x[k],y[k])
                    if settings['kind']=='tikhonov':errors.append(float(np.max(np.abs(x['start']-y['start']))))
            if errors:assert max(errors)<1e-8
            tc={'dataset':{'reference':file_reference(result/'dataset.json'),'used_ids':{k:nb['split_ids'][k] for k in ('train','validation')}},'geometry':asdict(g),
                'projection':{'kind':'block','operator_sha256':pc['operator_sha256']}}
            loader=CachedData(tc);batch=loader.batch(nb['split_ids']['train'],'cpu',split='train');assert batch['start'].shape==(2,1,8,8)
            out.append({'kind':settings['kind'],'run':result.name,'unchanged_measurements_and_targets':True,'training_loader_passed':True,'tikhonov_replay_error':max(errors) if errors else None})
        write_json(a.path/'summary.json',{'passed':True,'operator_run':oa.path.name,'data_run':data.name,'start_runs':out})
        print(json.dumps(out,indent=2),flush=True)
    return a.path
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);x=p.parse_args();run(x.root)
