"""Two-block row-space factorization, avoiding a single oversized SVD.

Uses every nonzero measurement row. The cutoff is applied per residual block,
not to the singular spectrum of the entire matrix. Produces a kernel-projector
approximation, not a general-data Moore-Penrose inverse.
"""
import time
import numpy as np
import torch
from dpnn_experiments.cuda_svd import XGesvd
from dpnn_experiments.projections import MatrixOperator
from dpnn_experiments.projection_diagnostics import array, dp_metrics, directional_check
from dpnn_experiments.provenance import write_json


def build_basis(archive,dense,options):
    active=torch.nonzero(torch.linalg.vector_norm(dense,dim=1)>0,as_tuple=True)[0]
    groups=torch.tensor_split(active,options['row_blocks'])
    np.savez_compressed(archive.path/'active_rows.npz',active_rows=array(active))
    basis=None;records=[]
    started=time.perf_counter()
    for index,rows in enumerate(groups):
        block=dense[rows].T
        # Preserve the first factorization on the original block. Subsequent
        # blocks are projected twice to reduce loss of orthogonality.
        original_norm=float(torch.linalg.norm(block))
        if basis is not None:
            for _ in range(2):block=block-basis@(basis.T@block)
        block=block.T.contiguous().T
        solver=XGesvd();t=time.perf_counter()
        try:u,s,vh=solver.factor(block,lambda item:archive.event('block_workspace',block=index,**item))
        finally:solver.close()
        factor_seconds=time.perf_counter()-t
        cutoff=options['block_svd_rtol']*float(s[0])
        rank=int((s>cutoff).sum())
        write_json(archive.path/f'block_{index}_summary.json',{'block':index,'rows':len(rows),'rank':rank,
            'cutoff':cutoff,'factor_seconds':factor_seconds,'original_block_frobenius_norm':original_norm})
        # Complete SVD outputs are retained, not just the kept columns.
        np.save(archive.path/f'block_{index}_U.npy',array(u),allow_pickle=False)
        np.save(archive.path/f'block_{index}_Vh.npy',array(vh),allow_pickle=False)
        np.savez_compressed(archive.path/f'block_{index}_spectrum.npz',singular=array(s),rows=array(rows))
        candidate=u[:,:rank].clone()
        del u,s,vh,block
        if basis is not None:
            for _ in range(2):candidate=candidate-basis@(basis.T@candidate)
        # Final QR only orthonormalizes retained vectors; no new rank decision.
        q,r=torch.linalg.qr(candidate,mode='reduced')
        np.save(archive.path/f'block_{index}_retained_QR_R.npy',array(r),allow_pickle=False)
        if basis is None:basis=q
        else:basis=torch.cat((basis,q),dim=1)
        record={'block':index,'rank_added':rank,'total_basis_columns':basis.shape[1],
                'factor_seconds':factor_seconds,'qr_min_abs_diagonal':float(torch.diagonal(r).abs().min())}
        records.append(record);archive.event('block_finished',**record)
        print(f'Block {index} finished: rank {rank}, total {basis.shape[1]}',flush=True)
        del candidate,q,r
        torch.cuda.empty_cache()
    torch.cuda.synchronize()
    seconds=time.perf_counter()-started
    np.save(archive.path/'row_space_basis.npy',array(basis),allow_pickle=False)
    return basis,records,seconds


def run(archive,config,device,radon):
    options=config['computation']
    if config['mode']=='block_reference':
        with np.load(config['dataset']['small_reference']['path'],allow_pickle=False) as z:
            matrix=z['matrix'].copy();expected=z['null'].copy();probes=z['probes'].copy()
        np.savez_compressed(archive.path/'reference_inputs.npz',matrix=matrix,null=expected,probes=probes)
        dense=torch.tensor(matrix,device=device)
        basis,blocks,seconds=build_basis(archive,dense,options)
        p=torch.eye(dense.shape[1],device=device,dtype=torch.float64)-basis@basis.T
        relative=float(np.linalg.norm(array(p)-expected)/np.linalg.norm(expected))
        result={'mode':'block_reference','projector_vs_global_svd_relative_error':relative,
                'basis_columns':basis.shape[1],'blocks':blocks,'seconds_with_factor_saving':seconds}
        write_json(archive.path/'summary.json',result)
        if relative>1e-8:raise ArithmeticError('Block row-space basis failed small global-SVD comparison')
        return
    op=MatrixOperator.whitened_radon(radon,device)
    dense=op.matrix.to_dense()
    basis,blocks,seconds=build_basis(archive,dense,options)
    del dense;torch.cuda.empty_cache()
    with np.load(config['dataset']['probes']['path'],allow_pickle=False) as z:probes=z['probes'].copy()
    np.savez_compressed(archive.path/'inputs.npz',probes=probes)
    values=torch.tensor(probes,device=device)
    project=lambda x:x-basis@(basis.T@x)
    records=[]
    for j,name in enumerate(config['dataset']['probe_names']):
        v=values[:,j];project(v);torch.cuda.synchronize();times=[]
        for _ in range(options['timing_repetitions']):
            start=time.perf_counter();null=project(v);torch.cuda.synchronize();times.append(time.perf_counter()-start)
        twice=project(null);q=v-null
        record={'probe':name,'idempotence_defect_norm':float(torch.linalg.norm(twice-null)),
                'orthogonality_inner_product':float(torch.dot(null,q)),
                'projection_seconds':times,**dp_metrics(op,v,null,options['beta_fractions'])}
        records.append(record);archive.event('block_projection_checked',**record)
        np.savez_compressed(archive.path/f'projection_{name}.npz',null=array(null),q=array(q),twice=array(twice))
    rng=np.random.Generator(np.random.PCG64(config['randomness']['seed']+4))
    coefficients=torch.tensor(rng.normal(size=(basis.shape[1],3)),device=device)
    ortho=float(torch.linalg.norm(basis.T@(basis@coefficients)-coefficients)/torch.linalg.norm(coefficients))
    np.savez_compressed(archive.path/'orthogonality_probe.npz',coefficients=array(coefficients))
    direction=torch.tensor(rng.normal(size=op.shape[1]),device=device);direction/=torch.linalg.norm(direction)
    upstream=torch.tensor(rng.normal(size=op.shape[1]),device=device);upstream/=torch.linalg.norm(upstream)
    gradient=directional_check(archive,config,project,values[:,0],upstream,direction,'block_null')
    result={'mode':'block_factorization','basis_columns':basis.shape[1],'blocks':blocks,
            'seconds_including_block_saves':seconds,'random_orthogonality_relative_error':ortho,
            'projections':records,'gradient':gradient,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'limitations':['Block-local cutoff, not global singular-value truncation',
                           'No full 256-square SVD reference or exact-rank certificate',
                           'Produces a row-space basis and fixed projector; no general-data pseudoinverse supplied']}
    write_json(archive.path/'summary.json',result)
    if ortho>1e-8 or max(r['idempotence_defect_norm'] for r in records)>1e-8:
        raise ArithmeticError('Block basis orthogonality/idempotence check failed')
