"""Full-matrix residual checks and application benchmark of archived factors."""
import time
import numpy as np
import torch
from dpnn_experiments.projections import MatrixOperator
from dpnn_experiments.projection_diagnostics import array
from dpnn_experiments.provenance import write_json


def run(archive,config,device,radon):
    options=config['computation']
    reference=config['dataset']['factor_basis']
    started=time.perf_counter()
    basis=torch.tensor(np.load(reference['path'],allow_pickle=False),device=device)
    op=MatrixOperator.whitened_radon(radon,device)
    dense=op.matrix.to_dense()
    torch.cuda.synchronize()
    archive.event('verification_inputs_ready',seconds=time.perf_counter()-started,basis_shape=list(basis.shape))
    chunk=options['verification_chunk_rows']
    row_residuals=[];row_norms=[];timings=[]
    t=time.perf_counter()
    for start in range(0,dense.shape[0],chunk):
        stop=min(start+chunk,dense.shape[0])
        block=dense[start:stop]
        residual=block-(block@basis)@basis.T
        row_residuals.extend(array(torch.linalg.vector_norm(residual,dim=1)).tolist())
        row_norms.extend(array(torch.linalg.vector_norm(block,dim=1)).tolist())
        timings.append({'row_start':start,'row_stop':stop,'elapsed_seconds':time.perf_counter()-t})
        archive.event('full_matrix_residual_chunk',**timings[-1])
    leak=float(np.linalg.norm(row_residuals));matrix_norm=float(np.linalg.norm(row_norms))
    np.savez_compressed(archive.path/'full_matrix_row_residuals.npz',row_residual_norms=row_residuals,row_norms=row_norms)
    write_json(archive.path/'matrix_chunk_times.json',timings)
    del dense,residual,block
    orth_rows=[];orth_times=[];t=time.perf_counter()
    rank=basis.shape[1]
    for start in range(0,rank,chunk):
        stop=min(start+chunk,rank)
        residual=basis[:,start:stop].T@basis
        residual[torch.arange(stop-start,device=device),torch.arange(start,stop,device=device)]-=1
        orth_rows.extend(array(torch.linalg.vector_norm(residual,dim=1)).tolist())
        orth_times.append({'row_start':start,'row_stop':stop,'elapsed_seconds':time.perf_counter()-t})
        archive.event('full_orthogonality_chunk',**orth_times[-1])
    orth=float(np.linalg.norm(orth_rows))
    np.savez_compressed(archive.path/'full_orthogonality_rows.npz',row_residual_norms=orth_rows)
    write_json(archive.path/'orthogonality_chunk_times.json',orth_times)
    del residual
    torch.cuda.empty_cache()
    project=lambda x:x-basis@(basis.T@x)
    rng=np.random.Generator(np.random.PCG64(config['randomness']['seed']+5))
    batches=[]
    for count in options['batch_sizes']:
        values=rng.normal(size=(op.shape[1],count));values/=np.linalg.norm(values,axis=0)
        upstream=rng.normal(size=values.shape);upstream/=np.linalg.norm(upstream,axis=0)
        x=torch.tensor(values,device=device,requires_grad=True)
        w=torch.tensor(upstream,device=device)
        project(x);torch.cuda.synchronize()
        forwards=[];backwards=[]
        for _ in range(options['timing_repetitions']):
            t=time.perf_counter();out=project(x);torch.cuda.synchronize();forwards.append(time.perf_counter()-t)
            t=time.perf_counter();grad=torch.autograd.grad((out*w).sum(),x)[0];torch.cuda.synchronize();backwards.append(time.perf_counter()-t)
        expected=project(w)
        relative=float(torch.linalg.norm(grad-expected)/torch.linalg.norm(expected))
        np.savez_compressed(archive.path/f'batch_{count}.npz',inputs=values,upstream=upstream,output=array(out),gradient=array(grad))
        batches.append({'batch_size':count,'forward_seconds':forwards,'backward_seconds':backwards,
                        'gradient_against_fixed_projector_relative_error':relative})
        if relative>1e-10:raise ArithmeticError('Fixed-factor batch gradient check failed')
    with np.load(config['dataset']['probes']['path'],allow_pickle=False) as z:probes=z['probes'].copy()
    reference_null=array(project(torch.tensor(probes,device=device)))
    np.savez_compressed(archive.path/'reference_projections.npz',probes=probes,null=reference_null)
    comparisons=[]
    for item in config['dataset'].get('comparison_outputs',[]):
        with np.load(item['path'],allow_pickle=False) as z:other=z['null'].copy()
        if other.ndim==1:
            j=config['dataset']['probe_names'].index(item['probe'])
            errors=[float(np.linalg.norm(other-reference_null[:,j]))]
        else:
            errors=[float(np.linalg.norm(other[:,j]-reference_null[:,j])) for j in range(other.shape[1])]
        comparisons.append({'name':item['name'],'probe':item.get('probe','both'),'distance_to_block_projector_per_unit_input':errors})
    result={'mode':'factor_verification','basis_columns':rank,'matrix_frobenius_norm':matrix_norm,
            'MP_frobenius_norm':leak,'MP_relative_frobenius_norm':leak/matrix_norm,
            'QtQ_minus_I_frobenius_norm':orth,'derived_P_idempotence_operator_bound':(1+orth)*orth,
            'batches':batches,'comparisons':comparisons,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'limitations':['Full-matrix Float64 residual measurements, not interval-arithmetic certificates',
                           'Block projector is the comparison reference, not a proven exact full-SVD projector',
                           'No general-data pseudoinverse or network training benchmark']}
    write_json(archive.path/'summary.json',result)
    print(result,flush=True)
    if orth>1e-8:raise ArithmeticError('Global basis orthogonality check failed')
