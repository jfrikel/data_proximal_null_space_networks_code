# SPDX-License-Identifier: GPL-3.0-only
"""Archived solver, projection and standalone evaluator qualification."""
import argparse,copy,json
from pathlib import Path
import numpy as np
import torch
from dpnn_experiments.reconstruction import reconstruct, gradient, gradient_adjoint
from dpnn_experiments.network_layers import FixedBlockProjection, ScalarDataProximalNetwork, NullSpaceNetwork
from dpnn_experiments.projections import MatrixOperator
from dpnn_experiments.projection_acceptance import accept_projection
from dpnn_experiments.training import build_model
from dpnn_experiments.training_archive import TrainingArchive, file_reference
from dpnn_experiments.provenance import write_json
from dpnn_experiments.evaluator import execute, protocol, ratio, METRICS

CROOT=Path(__file__).resolve().parents[1]


def run(root):
    root=Path(root)
    config={'schema_version':1,'purpose':'Technical qualification only: independent analytic TV/Tikhonov, SVD projection, evaluator split/guards',
      'dataset':{'kind':'synthetic technical fixture, separate from pilot data'},'geometry':{'size':8},
      'randomness':{'seed':3601},'computation':{'device':'cpu','dtype':'float64'},'training':{'applicable':False}}
    with TrainingArchive(config,root/'runs',CROOT,root/'metadata') as a:
        torch.set_num_threads(4);torch.manual_seed(3601);rng=np.random.default_rng(3601);tests={}
        x=torch.randn(4,4,dtype=torch.float64);dual=torch.randn(4,4,2,dtype=torch.float64)
        tests['TV_adjoint_error']=float(abs((gradient(x)*dual).sum()-(x*gradient_adjoint(dual)).sum()))
        assert tests['TV_adjoint_error']<1e-12
        tv={'kind':'tv','alpha':.1,'rtol':1e-9,'atol':1e-11,'max_iterations':20000,'check_every':10,'step_safety':.99}
        # Identity denoising, 2x2: two equal rows [0,1]; exact shrinkage [alpha,1-alpha]. h=1.
        op=MatrixOperator.dense(np.eye(4),'cpu');y=torch.tensor([0.,1.,0.,1.],dtype=torch.float64)
        sol,info=reconstruct(op,y,2,tv);write_json(a.path/'tv_analytic_solver.json',info)
        tests['TV_analytic_error']=float((sol-torch.tensor([[.1,.9],[.1,.9]],dtype=torch.float64)).abs().max())
        assert info['passed'] and tests['TV_analytic_error']<1e-7
        # Constant and zero null cases, and insufficient iteration rejection.
        for name,val in [('zero',0.),('constant',.3)]:
            result,trace=reconstruct(op,torch.full((4,),val,dtype=torch.float64),2,tv)
            write_json(a.path/f'tv_{name}.json',trace);assert trace['passed'] and float((result-val).abs().max())<1e-7
        _,trace=reconstruct(op,y,2,{**tv,'max_iterations':1});assert not trace['passed'];tests['solver_limit_rejected']=True
        mat=rng.normal(size=(10,16));op=MatrixOperator.dense(mat,'cpu');y=torch.randn(10,dtype=torch.float64)
        tik={'kind':'tikhonov','alpha':.01,'rtol':1e-10,'atol':1e-12,'max_iterations':200}
        sol,info=reconstruct(op,y,4,tik);u,s,v=np.linalg.svd(mat,full_matrices=False)
        ref=(v.T@(s/(s*s+.01)*(u.T@y.numpy()))).reshape(4,4)/.5
        tests['tikhonov_SVD_error']=float(np.max(np.abs(sol.numpy()-ref)));assert info['passed'] and tests['tikhonov_SVD_error']<1e-8
        write_json(a.path/'tikhonov_reference.json',info)
        # A genuine low-rank fixture checks projection against independently formed pseudoinverse.
        u,_=np.linalg.qr(rng.normal(size=(12,6)));v,_=np.linalg.qr(rng.normal(size=(64,6)));mat=(u*np.geomspace(2,.05,6))@v.T
        proj=FixedBlockProjection(torch.tensor(v),torch.tensor(mat),coordinates='physical',image_weight=.0625,contract={'fixture':True})
        accept_projection(proj,a)
        xx=torch.randn(2,64,dtype=torch.float64,requires_grad=True);reference=np.eye(64)-np.linalg.pinv(mat,rcond=1e-12)@mat
        tests['projection_SVD_error']=float(np.max(np.abs(proj(xx).detach().numpy()-xx.detach().numpy()@reference.T)))
        assert tests['projection_SVD_error']<1e-10 and torch.autograd.gradcheck(proj,(xx,),eps=1e-6,atol=1e-6,rtol=1e-4)
        correction=torch.nn.Linear(64,64,bias=False).double();dp=ScalarDataProximalNetwork(correction,proj);ns=NullSpaceNetwork(correction,proj)
        assert torch.equal(dp(xx,0.),ns(xx));assert torch.allclose(dp(xx,1e9),xx+correction(xx),atol=1e-12,rtol=1e-12)
        tests['DP_endpoint_checks']=True;tests['projection_gradcheck']=True
        assert ratio(0,0) is None
        # Full evaluator plumbing on a disjoint fixture, including locked test split.
        size=8;mat=np.eye(64);mat[-4:]=0;basis=np.eye(64)[:,:60]
        projection_file=a.path/'projection_fixture.npz';np.savez_compressed(projection_file,matrix=mat,basis=basis,image_weight=.0625)
        g={'size':8,'angles':64,'detectors':1,'angle_min_deg':-60.,'angle_max_deg':60.,'half_width':2**.5}
        splits={k:[f'eval-fixture/{k}/0'] for k in ('train','validation','test')};records=[]
        for split,names in splits.items():
            name=names[0];target=rng.uniform(.1,.8,(1,8,8));clean=mat@target.ravel()*.25;noise=rng.normal(size=64);noise*=.01/np.linalg.norm(noise)
            path=a.path/f'{split}.npz';np.savez_compressed(path,target=target,measurement=clean+noise,analytic_clean=clean,raster_clean=clean,delta=.01)
            records.append({'id':name,'split':split,'file':file_reference(path),'noise_realization_id':name+'/noise0'})
        binding=a.path/'dataset.json';write_json(binding,{'geometry':g,'split_ids':splits,'records':records})
        mc=json.loads((CROOT/'configs/tests/training_qualification.json').read_text());mc['geometry']=g;mc['projection']={'kind':'fixture','reference':file_reference(projection_file)}
        mc['model']['dropout']=0.;mc['dataset']['used_ids']={k:splits[k] for k in ('train','validation')};mc['dataset']['reference']=file_reference(binding)
        entries=[]
        for method in ('RES','NSN','DP','DP_TWO'):
            source=copy.deepcopy(mc);source['model']['method']=method;source['purpose']='Untrained seeded evaluator plumbing fixture, not scientific training'
            with TrainingArchive(source,root/'runs',CROOT,root/'metadata') as source_archive:
                model=build_model(source,torch.device('cpu'),source_archive);weight=source_archive.weights(model,0)
            entries.append({'method':method,'training_manifest':file_reference(source_archive.path/'manifest.json'),'weights':weight,'trained_start':'tikhonov'})
        ec={'schema_version':1,'purpose':'Standalone evaluator fixture; both starts, three wrappers, beta endpoints; not scientific test data',
            'dataset':{'reference':file_reference(binding)},'geometry':g,'randomness':{'seed':3601},'computation':{'device':'cpu','dtype':'float64','cpu_threads':4},
            'model_config':mc,'models':entries,'starts':[tik,{**tv,'alpha':.01,'rtol':1e-6,'atol':1e-9}],
            'beta_multipliers':[0.,.5,1.,4.],'split':'validation','ids':splits['validation'],'diagnostic_ids':splits['validation'],
            'allow_start_transfer':True,'metrics':{**METRICS,'ssim_window':7},'tolerances':{'atol':1e-12,'rtol':1e-10},'frozen_protocol':None}
        validation=execute(ec,root)
        tc=copy.deepcopy(ec);tc.update(split='test',ids=splits['test'],diagnostic_ids=splits['test'])
        locked=a.path/'frozen_test.json';write_json(locked,{'protocol':protocol(tc),'selection_closed':True});tc['frozen_protocol']=file_reference(locked)
        test=execute(tc,root)
        # Manifested expected guard failures. They must fail before numerical inference.
        guards=[]
        for name in ('unlocked_test','altered_test','wrong_split','wrong_weight','solver_limit','start_transfer','duplicate_model'):
            bad=copy.deepcopy(tc if name in ('unlocked_test','altered_test') else ec)
            if name=='unlocked_test':bad['frozen_protocol']=None
            elif name=='altered_test':bad['beta_multipliers']=[.75]
            elif name=='wrong_split':bad['ids']=splits['test'];bad['diagnostic_ids']=[]
            elif name=='wrong_weight':bad['models'][0]['weights']['sha256']='0'*64
            elif name=='start_transfer':bad['allow_start_transfer']=False
            elif name=='duplicate_model':bad['models'].append(copy.deepcopy(bad['models'][0]))
            else:bad['starts'][1]['max_iterations']=1
            before=set((root/'runs').iterdir())
            try:execute(bad,root)
            except (ValueError,ArithmeticError) as e:guards.append({'case':name,'error':str(e),'runs':[p.name for p in set((root/'runs').iterdir())-before]})
            else:raise AssertionError('Guard failed: '+name)
        # Routing fixture: six explicitly labeled untrained states, one per method/start.
        mapped=copy.deepcopy(ec);mapped['allow_start_transfer']=False
        mapped['models'] += [{**copy.deepcopy(e),'trained_start':'tv'} for e in entries]
        mapped['purpose']='Untrained fixture checks method/start routing only; no claim of TV training'
        routed=execute(mapped,root)
        routed_rows=[json.loads(x) for x in (routed/'metrics.jsonl').read_text().splitlines()]
        assert len(routed_rows)==22 and not any(x['start_transfer'] for x in routed_rows)
        tests['matched_start_routing_run']=routed.name
        tests.update(validation_run=validation.name,test_fixture_run=test.name,guards=guards,passed=True)
        write_json(a.path/'summary.json',tests);print(json.dumps(tests,indent=2),flush=True)
    return a.path

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);x=p.parse_args();run(x.root)
