# SPDX-License-Identifier: GPL-3.0-only
"""Archived fixed-factor NSN/scalar-DP integration and derivative diagnostics."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.func import functional_call

from tools.factorization_diagnostics import WorkerArchive, supervise
from dpnn_experiments.network_layers import FixedBlockProjection, NullSpaceNetwork, ScalarDataProximalNetwork, checked_archive_file
from dpnn_experiments.projection_diagnostics import setup, array, require
from dpnn_experiments.provenance import RunArchive, digest, write_json

SOURCE_ROOT = Path(__file__).resolve().parents[1]


class SmoothCorrection(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(size, size, dtype=torch.float64)/math.sqrt(size))
        self.bias = nn.Parameter(torch.randn(size, dtype=torch.float64)*0.1)

    def forward(self, z):
        return torch.tanh(z@self.weight.T+self.bias)


class PointwiseCorrection(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.2, dtype=torch.float64))
        self.bias = nn.Parameter(torch.tensor(0., dtype=torch.float64))

    def forward(self, z):
        return self.scale*torch.tanh(z)+self.bias


def checkpoint(archive, model, name):
    path = archive.path/(name+'.pt')
    torch.save(model.state_dict(), path)
    record = {'name': name, 'path': path.name, 'sha256': digest(path),
              'parameter_count': sum(p.numel() for p in model.parameters()),
              'training_steps': 0, 'state': 'initialized diagnostic weights; not trained'}
    manifest = json.loads((archive.path/'manifest.json').read_text())
    manifest.setdefault('evaluated_checkpoints', []).append(record)
    write_json(archive.path/'manifest.json', manifest)
    archive.event('checkpoint_before_evaluation', **record)
    return record


def save_arrays(archive, name, **values):
    np.savez_compressed(archive.path/(name+'.npz'), **{
        k: array(v) if isinstance(v, torch.Tensor) else v for k, v in values.items()})


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def proximity(archive, name, model, z, beta, config):
    with torch.no_grad():
        out, d = model(z, beta, return_details=True)
        projection = model.projection
        actual_data = projection.data(out-z)
        av = projection.data(d['value'])
        an = projection.data(d['null'])
        leaked = torch.linalg.vector_norm(an, dim=1)
        actual = torch.linalg.vector_norm(actual_data, dim=1)
        bound = d['beta']+(1-d['control'])*leaked
        decomposition = d['control'][:,None]*av+(1-d['control'][:,None])*an
        # Baseline residual uses fixed archived synthetic measurement perturbations.
        generator = np.random.Generator(np.random.PCG64(config['randomness']['seed']+7))
        noise = torch.tensor(generator.normal(size=tuple(av.shape)), device=z.device)
        noise = 0.01*noise/torch.linalg.vector_norm(noise, dim=1, keepdim=True)
        measurements = projection.data(z)+noise
        initial_residual = torch.linalg.vector_norm(projection.data(z)-measurements, dim=1)
        output_residual = torch.linalg.vector_norm(projection.data(out)-measurements, dim=1)
        allowed = config['computation']['proximity_atol']*torch.maximum(
            torch.ones_like(actual), torch.linalg.vector_norm(av, dim=1))
        save_arrays(archive, name, z=z, beta=d['beta'], output=out, control=d['control'],
                    correction=d['value'], null=d['null'], range=d['range'],
                    actual_data_change=actual_data, null_data=an, measurements=measurements)
        record = {'name': name, 'beta': array(d['beta']).tolist(), 'control': array(d['control']).tolist(),
                  'data_change_norm': array(actual).tolist(), 'null_leak_norm': array(leaked).tolist(),
                  'bound_with_leak': array(bound).tolist(),
                  'signed_excess_over_beta': array(actual-d['beta']).tolist(),
                  'roundoff_acceptance': array(allowed).tolist(),
                  'initial_residual': array(initial_residual).tolist(),
                  'output_residual': array(output_residual).tolist(),
                  'decomposition_absolute_error': float(torch.linalg.norm(actual_data-decomposition))}
        archive.event('proximity', **record)
        require(bool((actual <= bound+allowed).all()), 'Per-sample DP leakage bound failed')
        require(bool((output_residual <= initial_residual+bound+allowed).all()), 'Residual triangle bound failed')
        nsn = NullSpaceNetwork(model.correction, projection)
        zero = model(z, 0.)
        require(torch.equal(zero, nsn(z)), 'DP beta=0 differs from NSN')
        full = model(z, 2*torch.linalg.vector_norm(av, dim=1)+1)
        require(bool(torch.allclose(full, z+d['value'], atol=1e-11, rtol=1e-11)), 'Unclipped DP differs from RES')
        return record


def directional(archive, name, model, z, beta, config, rng):
    """Input and complete-parameter VJPs versus independent central differences."""
    z = z.detach().clone().requires_grad_()
    parameters = dict(model.correction.named_parameters())
    direction = torch.tensor(rng.normal(size=tuple(z.shape)), device=z.device)
    direction /= torch.linalg.vector_norm(direction)
    upstream = torch.tensor(rng.normal(size=tuple(z.shape)), device=z.device)
    upstream /= torch.linalg.vector_norm(upstream)
    directions = {key: torch.tensor(rng.normal(size=tuple(p.shape)), device=p.device)
                  for key, p in parameters.items()}
    length = torch.sqrt(sum((v*v).sum() for v in directions.values()))
    directions = {key: v/length for key,v in directions.items()}
    function = (lambda x: model(x)) if beta is None else (lambda x: model(x, beta))
    out = function(z)
    gradients = torch.autograd.grad((out*upstream).sum(), (z,)+tuple(parameters.values()))
    exact_x = float((gradients[0]*direction).sum())
    exact_p = float(sum((g*directions[k]).sum() for k,g in zip(parameters, gradients[1:])))
    save_arrays(archive, name+'_inputs', z=z, direction=direction, upstream=upstream,
                input_gradient=gradients[0], beta=np.array([]) if beta is None else beta)
    torch.save({'directions': {k:v.cpu() for k,v in directions.items()},
                'gradients': {k:g.cpu() for k,g in zip(parameters, gradients[1:])}},
               archive.path/(name+'_parameter_derivatives.pt'))
    rows=[]
    with torch.no_grad():
        epsilons = config['computation'].get('unet_directional_epsilons',
                    config['computation']['directional_epsilons']) if name.startswith('unet') else config['computation']['directional_epsilons']
        for eps in epsilons:
            fd_x = float(((function(z+eps*direction)-function(z-eps*direction))*upstream).sum()/(2*eps))
            params_plus = {'correction.'+k:p+eps*directions[k] for k,p in parameters.items()}
            params_minus = {'correction.'+k:p-eps*directions[k] for k,p in parameters.items()}
            args = (z,) if beta is None else (z,beta)
            plus = functional_call(model, params_plus, args)
            minus = functional_call(model, params_minus, args)
            fd_p = float(((plus-minus)*upstream).sum()/(2*eps))
            rows.append({'epsilon':eps, 'input_autograd':exact_x, 'input_finite_difference':fd_x,
                         'input_absolute_error':abs(exact_x-fd_x),
                         'parameter_autograd':exact_p, 'parameter_finite_difference':fd_p,
                         'parameter_absolute_error':abs(exact_p-fd_p)})
    smooth = not name.startswith('unet')
    atol = config['computation']['directional_atol']
    rtol = config['computation']['smooth_directional_rtol' if smooth else 'relu_directional_rtol']
    passed = any(r['input_absolute_error'] <= atol+rtol*abs(exact_x)
                 and r['parameter_absolute_error'] <= atol+rtol*abs(exact_p) for r in rows)
    record = {'name':name,'passed':passed,'atol':atol,'rtol':rtol,'checks':rows,
              'note':'Fixed beta; ReLU network may cross activation kinks; three declared step sizes'}
    write_json(archive.path/(name+'_gradient.json'), record)
    archive.event('directional_gradients', **record)
    require(passed, 'Layer input/parameter directional derivative failed')
    require(all(b.grad is None and not b.requires_grad for b in model.projection.buffers()),
            'Fixed operator/factor unexpectedly received gradients')
    return record


def analytic_vjp(archive, name, model, z, beta, rng):
    """Independent analytic layer adjoint, then the network's selected VJP.

This compares the full chain rule even at a network kink; it does not assert
classical differentiability of ReLU/MaxPool there. Factors are used directly.
"""
    z=z.detach().clone().requires_grad_()
    parameters=dict(model.correction.named_parameters())
    w=torch.tensor(rng.normal(size=tuple(z.shape)),device=z.device)
    w/=torch.linalg.vector_norm(w)
    output=model(z) if beta is None else model(z,beta)
    observed=torch.autograd.grad((output*w).sum(),(z,)+tuple(parameters.values()))
    value=model.correction(z)
    p=model.projection
    with torch.no_grad():
        v=value.reshape(z.shape[0],-1).T
        upstream=w.reshape(z.shape[0],-1).T
        q=p.basis@(p.basis.T@v)
        rw=p.basis@(p.basis.T@upstream)
        gw=upstream-rw
        if beta is not None:
            cv=p.data_scale*(torch.sparse.mm(p.matrix,v) if p.matrix.is_sparse else p.matrix@v)
            r=torch.linalg.vector_norm(cv,dim=0)
            budget=torch.as_tensor(beta,device=z.device,dtype=torch.float64).expand_as(r)
            active=(budget>0)&(r>budget)
            safe=torch.where(active,r,torch.ones_like(r))
            t=torch.where(budget==0,torch.zeros_like(r),torch.where(active,budget/safe,torch.ones_like(r)))
            transpose=p.matrix.transpose(0,1)
            ctc=p.data_scale*(torch.sparse.mm(transpose,cv) if p.matrix.is_sparse else transpose@cv)
            dt=torch.where(active,-budget/safe**3,torch.zeros_like(r))[None,:]*ctc
            gw=gw+t[None,:]*rw+(q*upstream).sum(dim=0,keepdim=True)*dt
        layer_upstream=gw.T.reshape_as(value)
    propagated=torch.autograd.grad((value*layer_upstream).sum(),(z,)+tuple(parameters.values()))
    expected=(propagated[0]+w,)+propagated[1:]
    rows=[]
    for key,a,b in zip(['input']+list(parameters),observed,expected):
        absolute=float(torch.linalg.vector_norm(a-b))
        scale=float(torch.linalg.vector_norm(b))
        rows.append({'tensor':key,'absolute_error':absolute,'reference_norm':scale,
                     'relative_error':absolute/max(scale,1e-30)})
        require(absolute<=1e-11+1e-10*scale,'Independent layer chain rule mismatch')
    save_arrays(archive,name+'_analytic_vjp_inputs',z=z,upstream=w,layer_upstream=layer_upstream,
                observed_input_gradient=observed[0],expected_input_gradient=expected[0],
                beta=np.array([]) if beta is None else beta)
    torch.save({'observed':{k:a.cpu() for k,a in zip(parameters,observed[1:])},
                'expected':{k:a.cpu() for k,a in zip(parameters,expected[1:])}},
               archive.path/(name+'_analytic_parameter_vjp.pt'))
    record={'name':name,'passed':True,'tensors':rows,
            'meaning':'Independent analytic layer adjoint composed with selected network VJP; no differentiability assertion at network kinks'}
    write_json(archive.path/(name+'_analytic_vjp.json'),record)
    archive.event('analytic_vjp',name=name,max_absolute_error=max(r['absolute_error'] for r in rows))
    return record


def pooling_ties(archive,network,inputs):
    records=[];handles=[]
    def hook(name):
        def inspect(module,args,output):
            windows=args[0].unfold(2,2,2).unfold(3,2,2)
            top=windows.amax(dim=(-2,-1))
            count=(windows==top[...,None,None]).sum(dim=(-2,-1))
            ties=((top>0)&(count>1)).flatten(1).sum(dim=1)
            records.append({'module':name,'positive_maximum_ties_per_sample':array(ties).tolist(),
                            'windows_per_sample':top[0].numel()})
        return inspect
    try:
        for name,module in network.named_modules():
            if isinstance(module,nn.MaxPool2d):handles.append(module.register_forward_hook(hook(name)))
        with torch.no_grad():network(inputs)
    finally:
        for h in handles:h.remove()
    write_json(archive.path/'unet_pooling_ties.json',records)
    return records


def small(archive, config, device):
    rng = np.random.Generator(np.random.PCG64(config['randomness']['seed']))
    left,_ = np.linalg.qr(rng.normal(size=(6,4)))
    right,_ = np.linalg.qr(rng.normal(size=(9,4)))
    matrix = (left*np.array([2.,1.,.4,.1]))@right.T
    _,_,vh = np.linalg.svd(matrix, full_matrices=False)
    basis = vh[:4].T
    p = np.eye(9)-np.linalg.pinv(matrix, rcond=1e-12)@matrix
    wx,wy = .04,.13
    physical_matrix = matrix*math.sqrt(wx/wy)
    save_arrays(archive, 'reference', M=matrix, A=physical_matrix, basis=basis, P=p,
                image_weight=wx, data_weight=wy)
    projection = FixedBlockProjection(torch.tensor(basis,device=device),torch.tensor(matrix,device=device),
        coordinates='physical',image_weight=wx,contract={'kind':'recorded_small_svd','reference':'reference.npz'})
    correction = SmoothCorrection(9).to(device)
    dp = ScalarDataProximalNetwork(correction,projection)
    checkpoint(archive,dp,'smooth_initial_state')
    z = torch.tensor(rng.normal(size=(3,9)),device=device)
    with torch.no_grad():
        value = correction(z)
        r = torch.linalg.vector_norm(projection.data(value),dim=1)
        beta = r*torch.tensor([0.,.3,2.],device=device,dtype=torch.float64)
        expected_t = np.array([0.,.3,1.])
        null = array(value)@p.T
        expected = array(z)+null+expected_t[:,None]*(array(value)-null)
        actual = array(dp(z,beta))
        norm_error = float(np.linalg.norm(array(r)-np.sqrt(wy)*np.linalg.norm(array(value)@physical_matrix.T,axis=1)))
        reference_error = float(np.linalg.norm(actual-expected))
        require(reference_error<1e-11 and norm_error<1e-12,'Independent physical norm/SVD formula mismatch')
    record = proximity(archive,'small_mixed_beta',dp,z,beta,config)
    save_arrays(archive,'small_independent_expected',expected=expected,actual=actual)
    # Functional substitution tests gradients to inputs and actual learnable weights.
    x=z.detach().requires_grad_();w=correction.weight;b=correction.bias
    gradcheck = torch.autograd.gradcheck(
        lambda x,w,b: functional_call(dp,{'correction.weight':w,'correction.bias':b},(x,beta)),
        (x,w,b),eps=1e-6,atol=2e-7,rtol=2e-5)
    nsn=NullSpaceNetwork(correction,projection)
    gradcheck_nsn=torch.autograd.gradcheck(
        lambda x,w,b:functional_call(nsn,{'correction.weight':w,'correction.bias':b},(x,)),
        (x,w,b),eps=1e-6,atol=2e-7,rtol=2e-5)
    archive.event('full_gradchecks',dp=gradcheck,nsn=gradcheck_nsn)
    derivatives=[directional(archive,'smooth_dp',dp,z,beta,config,rng),
                 directional(archive,'smooth_nsn',nsn,z,None,config,rng)]
    # At the origin, beta=0 must also have the NSN derivative, not RES.
    identity=ScalarDataProximalNetwork(nn.Identity(),projection)
    origin=torch.zeros((1,9),dtype=torch.float64,device=device,requires_grad=True)
    jac0=torch.autograd.functional.jacobian(lambda x:identity(x,0.),origin).reshape(9,9)
    jacpositive=torch.autograd.functional.jacobian(lambda x:identity(x,1.),origin).reshape(9,9)
    require(np.linalg.norm(array(jac0)-(np.eye(9)+p))<1e-11,'Origin beta=0 derivative wrong')
    require(torch.equal(jacpositive,2*torch.eye(9,device=device,dtype=torch.float64)),'Positive-beta origin derivative wrong')
    save_arrays(archive,'origin_jacobians',zero_beta=jac0,positive_beta=jacpositive)
    # Coordinate conversion must preserve outputs and weighted beta values.
    white=FixedBlockProjection(projection.basis,projection.matrix,coordinates='whitened',image_weight=wx,
                              contract={'kind':'coordinate_reference'})
    physical_identity=identity(z,beta)
    white_identity=ScalarDataProximalNetwork(nn.Identity(),white)(z*math.sqrt(wx),beta)/math.sqrt(wx)
    require(torch.allclose(physical_identity,white_identity,atol=1e-12,rtol=1e-12),'Coordinate contract mismatch')
    cases=[]
    for name,m,q in [('zero_operator',np.zeros((2,3)),np.empty((3,0))),
                     ('full_column_rank',np.eye(3),np.eye(3))]:
        layer=FixedBlockProjection(torch.tensor(q,device=device),torch.tensor(m,device=device),
            coordinates='whitened',image_weight=1.,contract={'kind':name})
        model=ScalarDataProximalNetwork(nn.Identity(),layer)
        v=torch.tensor([[.2,-.3,.7]],device=device,dtype=torch.float64)
        cases.append(proximity(archive,name,model,v,0.,config))
    kink_projection=FixedBlockProjection(torch.tensor([[1.],[0.]],device=device,dtype=torch.float64),
        torch.tensor([[1.,0.]],device=device,dtype=torch.float64),coordinates='whitened',image_weight=1.,contract={'kind':'kink'})
    kink=ScalarDataProximalNetwork(nn.Identity(),kink_projection)
    a=torch.tensor([[1.,0.]],device=device,dtype=torch.float64,requires_grad=True);eps=1e-6
    middle=kink(a,1.)[0,0];slope=float(torch.autograd.grad(middle,a)[0][0,0])
    v=torch.tensor([[eps,0.]],device=device,dtype=torch.float64)
    left_slope=float((middle-kink(a-v,1.)[0,0])/eps)
    right_slope=float((kink(a+v,1.)[0,0]-middle)/eps)
    require(abs(slope-2)<1e-12 and abs(left_slope-2)<1e-8 and abs(right_slope-1)<1e-8,'Kink branch mismatch')
    guards=[]
    bad_state=copy.deepcopy(dp.state_dict());bad_state['projection._extra_state']['coordinates']='wrong'
    missing_state=copy.deepcopy(dp.state_dict());del missing_state['projection._extra_state']
    for name,call in [
        ('negative_beta',lambda:dp(z,-1.)),('nan_beta',lambda:dp(z,float('nan'))),
        ('infinite_beta',lambda:dp(z,float('inf'))),('wrong_beta_batch',lambda:dp(z,torch.ones(2))),
        ('trainable_beta',lambda:dp(z,torch.tensor(1.,requires_grad=True))),
        ('float32_input',lambda:dp(z.float(),1.)),('float32_conversion',lambda:projection.float()),
        ('empty_batch',lambda:dp(z[:0],1.)),('checkpoint_mismatch',lambda:dp.load_state_dict(bad_state)),
        ('checkpoint_missing_contract',lambda:dp.load_state_dict(missing_state,strict=False))]:
        try:call()
        except ValueError as error:guards.append({'name':name,'rejected':str(error)})
        else:raise ArithmeticError('Missing guard: '+name)
    state=torch.load(archive.path/'smooth_initial_state.pt',map_location=device,weights_only=True)
    dp.load_state_dict(state)
    require(torch.equal(dp(z,beta),torch.tensor(actual,device=device)),'Checkpoint replay differs')
    require(set(projection.state_dict())=={'_extra_state'},'Large buffers were serialized into checkpoint')
    result={'mode':'small','reference_absolute_error':reference_error,'weighted_norm_error':norm_error,
            'dp_gradcheck':gradcheck,'nsn_gradcheck':gradcheck_nsn,'gradients':derivatives,
            'proximity':[record]+cases,'guards':guards,
            'kink':{'autograd':slope,'left':left_slope,'right':right_slope},
            'origin_beta_zero_jacobian_error':float(np.linalg.norm(array(jac0)-(np.eye(9)+p))),
            'checkpoint_replay_exact':True,'large_buffers_excluded_from_checkpoint':True}
    write_json(archive.path/'summary.json',result)
    print(json.dumps(result),flush=True)


def load_fixed_projection(archive,config,device,root):
    settings=config['factors'];started=time.perf_counter()
    projection=FixedBlockProjection.from_archive(root/settings['factor_run'],
        basis_sha256=settings['basis_sha256'],operator_sha256=settings['operator_sha256'],
        geometry=config['geometry'],verification_run=root/settings['verification_run'],
        verification_summary_sha256=settings['verification_summary_sha256'],coordinates='physical',device=device)
    archive.event('verified_factor_loaded',seconds=time.perf_counter()-started,contract=projection.contract)
    write_json(archive.path/'projection_contract.json',projection.contract)
    return projection


def replay_gradients(archive,config,device,root):
    """Repeat the original failed directions and checkpoint, with smaller steps.

Input one-sided derivatives per sample distinguish a network kink from a
gradient error. No new random directions or acceptance tolerances are chosen.
"""
    projection=load_fixed_projection(archive,config,device,root)
    replay=config['replay_source'];source=root/replay['path']
    require(digest(source/'artifacts.json')==replay['artifacts_sha256'],'Replay archive index mismatch')
    files={name:checked_archive_file(source,name,sha) for name,sha in replay['files'].items()}
    from src.unet import UNet
    net=UNet(in_channels=1,out_channels=1,bilinear=True).double().to(device).eval()
    dp=ScalarDataProximalNetwork(net,projection)
    dp.load_state_dict(torch.load(files['unet_initial_state.pt'],map_location=device,weights_only=True))
    checkpoint(archive,dp,'replayed_unet_state')
    parameters=dict(net.named_parameters());records=[]
    for name in ('unet_dp','unet_nsn'):
        with np.load(files[name+'_inputs.npz'],allow_pickle=False) as data:
            z=torch.tensor(data['z'],device=device,requires_grad=True)
            direction=torch.tensor(data['direction'],device=device)
            upstream=torch.tensor(data['upstream'],device=device)
            old_input_gradient=torch.tensor(data['input_gradient'],device=device)
            beta=torch.tensor(data['beta'],device=device) if name=='unet_dp' else None
        old=torch.load(files[name+'_parameter_derivatives.pt'],map_location=device,weights_only=True)
        directions=old['directions']
        model=dp if beta is not None else NullSpaceNetwork(net,projection)
        args=(z,beta) if beta is not None else (z,)
        output=model(*args)
        grads=torch.autograd.grad((output*upstream).sum(),(z,)+tuple(parameters.values()))
        input_replay_error=float(torch.linalg.vector_norm(grads[0]-old_input_gradient))
        parameter_replay_errors={k:float(torch.linalg.vector_norm(g-old['gradients'][k]))
                                 for k,g in zip(parameters,grads[1:])}
        require(input_replay_error<1e-12 and max(parameter_replay_errors.values())<1e-12,
                'Original checkpoint/direction derivative did not replay')
        exact_x=float((grads[0]*direction).sum())
        exact_p=float(sum((g*directions[k]).sum() for k,g in zip(parameters,grads[1:])))
        save_arrays(archive,name+'_replay_inputs',z=z,direction=direction,upstream=upstream,
                    input_gradient=grads[0],beta=np.array([]) if beta is None else beta)
        torch.save({'gradients':{k:g.cpu() for k,g in zip(parameters,grads[1:])}},
                   archive.path/(name+'_replayed_parameter_gradients.pt'))
        rows=[];sided=[]
        with torch.no_grad():
            base=output.detach()
            function=(lambda x:model(x,beta)) if beta is not None else (lambda x:model(x))
            for eps in config['computation']['replay_epsilons']:
                plus=function(z+eps*direction);minus=function(z-eps*direction)
                fd_x=float(((plus-minus)*upstream).sum()/(2*eps))
                pplus=functional_call(model,{'correction.'+k:p+eps*directions[k] for k,p in parameters.items()},args)
                pminus=functional_call(model,{'correction.'+k:p-eps*directions[k] for k,p in parameters.items()},args)
                fd_p=float(((pplus-pminus)*upstream).sum()/(2*eps))
                rows.append({'epsilon':eps,'input_autograd':exact_x,'input_finite_difference':fd_x,
                    'input_absolute_error':abs(exact_x-fd_x),'parameter_autograd':exact_p,
                    'parameter_finite_difference':fd_p,'parameter_absolute_error':abs(exact_p-fd_p)})
                for i,label in enumerate(('gaussian','circle')):
                    left=float(((base[i]-minus[i])*upstream[i]).sum()/eps)
                    right=float(((plus[i]-base[i])*upstream[i]).sum()/eps)
                    sided.append({'epsilon':eps,'sample':label,'left':left,'right':right,
                                  'autograd':float((grads[0][i]*direction[i]).sum()),'slope_gap':abs(right-left)})
        atol=config['computation']['directional_atol'];rtol=config['computation']['relu_directional_rtol']
        passed=any(r['input_absolute_error']<=atol+rtol*abs(exact_x)
                   and r['parameter_absolute_error']<=atol+rtol*abs(exact_p) for r in rows)
        record={'name':name,'passed':passed,'atol':atol,'rtol':rtol,'checks':rows,
                'input_gradient_replay_error':input_replay_error,
                'parameter_gradient_replay_errors':parameter_replay_errors,'input_one_sided':sided}
        records.append(record);write_json(archive.path/(name+'_replay.json'),record)
        archive.event('original_direction_replay',name=name,passed=passed)
    result={'mode':'replay','source_run':source.name,'records':records,
            'meaning':'Same checkpoint, beta, inputs and directions as the failed run; smaller steps, original tolerances; one-sided input slopes at network kinks'}
    write_json(archive.path/'summary.json',result)
    print(json.dumps(result),flush=True)
    require(all(r['passed'] for r in records),'Original direction replay remains outside original tolerance')


def large(archive,config,device,root):
    projection=load_fixed_projection(archive,config,device,root)
    source=root/config['dataset']['probes']['path']
    if digest(source)!=config['dataset']['probes']['sha256']:raise ValueError('Probe hash mismatch')
    with np.load(source,allow_pickle=False) as data:base=data['probes'].T.copy()/math.sqrt(projection.image_weight)
    values=np.stack((base[0],base[1],np.zeros_like(base[0]),3*base[0])).reshape(4,1,256,256)
    z=torch.tensor(values,device=device)
    save_arrays(archive,'all_inputs',z=z)
    rng=np.random.Generator(np.random.PCG64(config['randomness']['seed']+3))
    results=[];gradients=[];timings=[]
    pointwise=PointwiseCorrection().to(device)
    dp=ScalarDataProximalNetwork(pointwise,projection)
    checkpoint(archive,dp,'pointwise_initial_state')
    with torch.no_grad():r=torch.linalg.vector_norm(projection.data(pointwise(z)),dim=1)
    beta=r*torch.tensor([0.,.25,0.,2.],device=device,dtype=torch.float64)
    results.append(proximity(archive,'pointwise_mixed_beta',dp,z,beta,config))
    gradients.append(directional(archive,'pointwise_dp',dp,z[:2],beta[:2],config,rng))
    gradients.append(directional(archive,'pointwise_nsn',NullSpaceNetwork(pointwise,projection),z[:2],None,config,rng))
    # Real existing U-Net, shared by NSN and DP; random fixed checkpoint, no optimizer.
    from src.unet import UNet
    unet=UNet(in_channels=1,out_channels=1,bilinear=True).double().to(device).eval()
    unet_dp=ScalarDataProximalNetwork(unet,projection)
    checkpoint(archive,unet_dp,'unet_initial_state')
    inputs=z[:2]
    ties=pooling_ties(archive,unet,inputs)
    with torch.no_grad():r=torch.linalg.vector_norm(projection.data(unet(inputs)),dim=1)
    beta=r*torch.tensor([.25,2.],device=device,dtype=torch.float64)
    results.append(proximity(archive,'unet_mixed_beta',unet_dp,inputs,beta,config))
    analytic=[analytic_vjp(archive,'unet_dp',unet_dp,inputs,beta,rng),
              analytic_vjp(archive,'unet_nsn',NullSpaceNetwork(unet,projection),inputs,None,rng)]
    # Finite differences use the generic Gaussian image; the piecewise-constant
    # circle may have positive MaxPool ties. Both images retain full VJP checks.
    gradients.append(directional(archive,'unet_dp',unet_dp,inputs[:1],beta[:1],config,rng))
    gradients.append(directional(archive,'unet_nsn',NullSpaceNetwork(unet,projection),inputs[:1],None,config,rng))
    for name,model,budget in [('NSN',NullSpaceNetwork(unet,projection),None),('scalar_DP',unet_dp,beta)]:
        samples=[]
        for i in range(config['computation']['timing_repetitions']+1):
            x=inputs.detach().clone().requires_grad_();sync(device);t=time.perf_counter()
            out=model(x) if budget is None else model(x,budget)
            sync(device);forward=time.perf_counter()-t;t=time.perf_counter()
            grads=torch.autograd.grad(out.square().mean(),(x,)+tuple(unet.parameters()))
            sync(device);backward=time.perf_counter()-t
            if i:samples.append({'forward_seconds':forward,'backward_seconds':backward})
            del out,grads,x
        timings.append({'method':name,'batch_size':2,'warmup':1,'samples':samples})
    # Native autocast around the projection must not silently change its arithmetic.
    with torch.no_grad():
        expected=projection(z)
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16):actual=projection(z)
    require(torch.equal(expected,actual) and actual.dtype==torch.float64,'Autocast changed fixed projection')
    result={'mode':'large','proximity':results,'gradients':gradients,'timings':timings,
            'analytic_vjp':analytic,'pooling_ties':ties,
            'autocast_projection_unchanged':True,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'limitations':['Fixed untrained U-Net; no optimizer/training or reconstruction-quality claim',
                           'Float64 only; projection has the archived blockwise numerical rank',
                           'General pseudoinverse and full training/resume remain separate tasks']}
    write_json(archive.path/'summary.json',result)
    print(json.dumps(result),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--worker',type=Path)
    args=parser.parse_args()
    if args.worker:
        archive=WorkerArchive(args.worker)
        config=json.loads((archive.path/'manifest.json').read_text())['configuration']
        device=setup(archive,config)
        if config['mode']=='small':small(archive,config,device)
        elif config['mode']=='large':large(archive,config,device,args.root)
        elif config['mode']=='replay':replay_gradients(archive,config,device,args.root)
        else:raise ValueError('Unknown diagnostic mode')
        return
    config=json.loads(args.config.read_text())
    os.environ['CUBLAS_WORKSPACE_CONFIG']=config['computation']['cublas_workspace_config']
    with RunArchive(config,args.root/'runs',SOURCE_ROOT,args.root/'metadata',
                    parent_run_id=config.get('factors',{}).get('factor_run','').split('/')[-1] or None) as archive:
        print('Run archive: '+str(archive.path),flush=True)
        supervise(archive,config,args.root,worker_module='tools.layer_diagnostics')


if __name__=='__main__':
    main()
