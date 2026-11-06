"""Worker functions for logged fixed-filter and row-preconditioned comparisons."""
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from dpnn_experiments.acceleration import FixedChebyshev, chebyshev_native, row_scaled_radon
from dpnn_experiments.projections import MatrixOperator, lsqr
from dpnn_experiments.projection_diagnostics import array, dp_metrics, directional_check, solve
from dpnn_experiments.provenance import write_json


def reference(archive,config,device):
    with np.load(config['dataset']['small_reference']['path'],allow_pickle=False) as z:
        matrix=z['matrix'].copy();singular=z['singular'].copy();vh=z['vh'].copy()
        expected_inverse=z['inverse'].copy();probes=z['probes'].copy()
    np.savez_compressed(archive.path/'reference_inputs.npz',matrix=matrix,singular=singular,vh=vh,
                        inverse=expected_inverse,probes=probes)
    op=MatrixOperator.dense(matrix,device)
    lower=op.norm_bound_squared*1e-6;upper=op.norm_bound_squared
    values=torch.tensor(probes,device=device)
    records=[]
    for k in [0,1,8,64,1024,4096]:
        layer=FixedChebyshev(op,k,lower,upper)
        actual=array(layer(values))
        # Independent scalar Chebyshev evaluation in the SVD basis.
        c=(upper+lower)/(upper-lower)
        z=(upper+lower-2*singular**2)/(upper-lower)
        numerator=np.polynomial.chebyshev.chebval(z,[0]*k+[1])
        denominator=np.polynomial.chebyshev.chebval(c,[0]*k+[1])
        gain=numerator/denominator
        expected=probes-vh.T@((1-gain)[:,None]*(vh@probes))
        relative=float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))
        records.append({'iterations':k,'spectral_comparison_relative_error':relative})
        np.savez_compressed(archive.path/f'chebyshev_reference_{k}.npz',actual=actual,expected=expected,gain=gain)
        if relative>1e-8:raise ArithmeticError('Chebyshev/SVD polynomial mismatch')
    # Compact dense rank-deficient case for a full Jacobian finite difference.
    small=np.array([[1.,2.,0.,0.,1.],[0.,1.,1.,2.,0.],[1.,3.,1.,2.,1.]])
    smallop=MatrixOperator.dense(small,device)
    x=torch.arange(10,dtype=torch.float64,device=device).reshape(5,2).requires_grad_(True)
    layer=FixedChebyshev(smallop,8,smallop.norm_bound_squared*1e-4)
    gradcheck=torch.autograd.gradcheck(layer,(x,),eps=1e-6,atol=1e-6,rtol=1e-4)
    custom=torch.autograd.grad(layer(x).sum(),x)[0]
    native=torch.autograd.grad(chebyshev_native(x,*layer.constants).sum(),x)[0]
    gradient_relative=float(torch.linalg.norm(custom-native)/torch.linalg.norm(native))
    np.savez_compressed(archive.path/'gradient_reference.npz',matrix=small,input=array(x),custom=array(custom),native=array(native))
    if gradient_relative>1e-10:raise ArithmeticError('Chebyshev backward differs from unrolled graph')
    rownorm=np.linalg.norm(matrix,axis=1);weights=np.ones_like(rownorm)
    np.divide(1.,rownorm,out=weights,where=rownorm>0)
    scaledop=MatrixOperator.dense(weights[:,None]*matrix,device)
    b=matrix@probes[:,0]
    settings=dict(config['computation']['lsqr']);settings.update(max_iterations=20000,normal_rtol=1e-13,data_rtol=1e-11)
    q,trace=lsqr(scaledop,torch.tensor(weights*b,device=device),require_data_residual=True,**settings)
    write_json(archive.path/'scaled_lsqr_trace.json',trace)
    relative=float(np.linalg.norm(array(q)-expected_inverse@b)/np.linalg.norm(expected_inverse@b))
    np.savez_compressed(archive.path/'scaled_lsqr_reference.npz',weights=weights,rhs=b,solution=array(q))
    if relative>1e-5 or not trace['converged']:raise ArithmeticError('Row-scaled LSQR SVD reference failed')
    write_json(archive.path/'summary.json',{'mode':'acceleration_reference','spectral_checks':records,
               'gradcheck':gradcheck,'gradient_relative_error':gradient_relative,
               'row_scaled_lsqr_image_error':relative,'row_scaled_lsqr_iterations':trace['iterations']})


def large(archive,config,device,radon):
    op=MatrixOperator.whitened_radon(radon,device)
    with np.load(config['dataset']['probes']['path'],allow_pickle=False) as z:probes=z['probes'].copy()
    np.savez_compressed(archive.path/'inputs.npz',probes=probes)
    values=torch.tensor(probes,device=device)
    records=[]
    for fraction in config['computation']['chebyshev_sigma_fractions']:
        for k in config['computation']['chebyshev_iterations']:
            layer=FixedChebyshev(op,k,op.norm_bound_squared*fraction**2)
            torch.cuda.synchronize();start=time.perf_counter()
            null=layer(values);torch.cuda.synchronize();elapsed=time.perf_counter()-start
            twice=layer(null)
            np.savez_compressed(archive.path/f'chebyshev_{fraction:.0e}_{k}.npz',null=array(null),twice=array(twice))
            for j,name in enumerate(config['dataset']['probe_names']):
                row={'method':'chebyshev','iterations':k,'lower_sigma_fraction_of_bound':fraction,
                     'probe':name,'projection_two_columns_seconds':elapsed,
                     'idempotence_defect_norm':float(torch.linalg.norm(twice[:,j]-null[:,j])),
                     **dp_metrics(op,values[:,j],null[:,j],config['computation']['beta_fractions'])}
                records.append(row);archive.event('chebyshev_checked',**row)
            write_json(archive.path/'chebyshev_results.json',records)
            print(f'Chebyshev fraction={fraction}, K={k}',flush=True)
    scaled,weights=row_scaled_radon(radon,device)
    np.savez_compressed(archive.path/'row_scaling.npz',weights=array(weights))
    write_json(archive.path/'row_scaling.json',{'minimum':float(weights.min()),'maximum':float(weights.max()),
               'norm_bound_squared':scaled.norm_bound_squared,'contract':'Solve D M q=D M v with zero start; exact consistent minimum-norm solution unchanged. General inconsistent least-squares objective is changed; no pseudoinverse equivalence claim.'})
    solvers=[]
    for j,name in enumerate(config['dataset']['probe_names']):
        v=values[:,j]
        q,trace=solve(archive,config,scaled,scaled.forward(v),f'scaled_lsqr_{name}',True)
        null=v-q
        q2,trace2=solve(archive,config,scaled,scaled.forward(null),f'scaled_lsqr_{name}_reprojection',True)
        row={'method':'row_scaled_lsqr','probe':name,'solver':trace,'reprojection_solver':trace2,
             'idempotence_defect_norm':float(torch.linalg.norm(q2)),**dp_metrics(op,v,null,config['computation']['beta_fractions'])}
        solvers.append(row)
        np.savez_compressed(archive.path/f'scaled_projection_{name}.npz',null=array(null),q=array(q),reprojected_range=array(q2))
        write_json(archive.path/'scaled_lsqr_results.json',solvers)
    rng=np.random.Generator(np.random.PCG64(config['randomness']['seed']+3))
    direction=torch.tensor(rng.normal(size=op.shape[1]),device=device);direction/=torch.linalg.norm(direction)
    upstream=torch.tensor(rng.normal(size=op.shape[1]),device=device);upstream/=torch.linalg.norm(upstream)
    layer=FixedChebyshev(op,1024,op.norm_bound_squared*1e-6)
    gradient=directional_check(archive,config,layer,values[:,0],upstream,direction,'chebyshev_1024')
    write_json(archive.path/'summary.json',{'mode':'acceleration','chebyshev':records,'row_scaled_lsqr':solvers,
               'gradient':gradient,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
               'limitations':['Finite filters are not exact null projectors','Scaled solver stopping uses scaled residuals; unscaled data leak also measured','Two diagnostic inputs; not a training benchmark']})
