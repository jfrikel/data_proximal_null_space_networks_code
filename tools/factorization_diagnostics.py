# SPDX-License-Identifier: GPL-3.0-only
"""Supervised, fully archived factorization diagnostics for a fixed Radon matrix.

Large factors stay in the canonical server archive. The parent records resource
samples and finalizes the archive even if a native solver times out or crashes.
"""
import argparse
import ctypes
from ctypes.util import find_library
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

import numpy as np

from dpnn_experiments.gpu_benchmark import load_operator, reference_metadata
from dpnn_experiments.provenance import RunArchive, digest, write_json, utc_now

SOURCE_ROOT = Path(__file__).resolve().parents[1]


class WorkerArchive:
    event = RunArchive.event

    def __init__(self, path):
        self.path = Path(path)
        self.run_id = self.path.name


def resources():
    sample = {"time_utc": utc_now(), "disk_free_bytes": shutil.disk_usage(Path.cwd()).free}
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu,power.draw",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
    sample.update(gpu_csv=result.stdout, gpu_query_returncode=result.returncode)
    if Path('/proc/meminfo').exists():
        sample['host_memory'] = Path('/proc/meminfo').read_text()
    return sample


def workspace_query(m, n):
    """Legacy workspace query used by the explicitly selected gesvd driver."""
    library_path = os.environ.get('DPNN_CUSOLVER_LIBRARY') or find_library('cusolver')
    if not library_path:
        raise RuntimeError('cuSOLVER not found; set DPNN_CUSOLVER_LIBRARY to your installed library path')
    library = ctypes.CDLL(library_path)
    handle = ctypes.c_void_p()
    library.cusolverDnCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    library.cusolverDnDestroy.argtypes = [ctypes.c_void_p]
    library.cusolverDnDgesvd_bufferSize.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                                  ctypes.POINTER(ctypes.c_int)]
    status = library.cusolverDnCreate(ctypes.byref(handle))
    if status:
        raise RuntimeError(f'cuSOLVER create status {status}')
    try:
        count = ctypes.c_int()
        status = library.cusolverDnDgesvd_bufferSize(handle, max(m,n), min(m,n), ctypes.byref(count))
        return {'api': 'cusolverDnDgesvd_bufferSize', 'status': status,
                'workspace_elements_int32': count.value,
                'workspace_bytes_if_valid': count.value*8 if status == 0 and count.value > 0 else None}
    finally:
        library.cusolverDnDestroy(handle)


def save_tensor(path, tensor):
    # Preserve all Float64 factors; no quantization or implicit truncation.
    np.save(path, tensor.detach().cpu().numpy(), allow_pickle=False)


def memory_record(torch):
    free, total = torch.cuda.mem_get_info()
    return {'free_bytes': free, 'total_bytes': total,
            'allocated_bytes': torch.cuda.memory_allocated(),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved()}


def measure(torch, function, repetitions):
    function()
    torch.cuda.synchronize()
    times = []
    for _ in range(repetitions):
        start = time.perf_counter()
        value = function()
        torch.cuda.synchronize()
        times.append(time.perf_counter()-start)
    return value, times


def run_worker(archive, config, root):
    import torch
    from dpnn_experiments.projection_diagnostics import setup, array, directional_check, dp_metrics
    from dpnn_experiments.projections import MatrixOperator
    device = setup(archive, config)
    torch.backends.cuda.preferred_linalg_library('cusolver')
    options = config['computation']
    mode = config['mode']
    ref = config['dataset']['reference_operator']
    radon = load_operator(root/'reference', ref)
    m,n = radon.shape
    preflight = {'shape': [m,n], 'nnz': len(radon.values),
                 'exact_zero_rows': int(m-len(np.unique(radon.rows))),
                 'dense_matrix_bytes': 8*m*n, 'reduced_factors_bytes': 8*(m*n+m*m+m),
                 'matrix_plus_factors_bytes': 8*(2*m*n+m*m+m),
                 'input_copy_bytes': 8*m*n, 'workspace_query': workspace_query(m,n),
                 'memory_before': memory_record(torch), 'resources': resources(),
                 'note': 'Workspace query excludes PyTorch copies, allocator reserve and internal library allocations.'}
    if options.get('svd_driver') == 'Xgesvd':
        from dpnn_experiments.cuda_svd import XGesvd
        solver=XGesvd(vectors=options.get('svd_vectors',True))
        try: preflight['workspace_query_64bit']=solver.query(max(m,n),min(m,n))
        finally: solver.close()
    if options.get('gram_driver') == 'Xsyevd':
        from dpnn_experiments.cuda_svd import Xsyevd
        solver=Xsyevd()
        try: preflight['gram_workspace_query_64bit']=solver.query(m)
        finally: solver.close()
    write_json(archive.path/'preflight.json', preflight)
    archive.event('preflight', **preflight)
    print(json.dumps(preflight), flush=True)
    if mode == 'preflight':
        return
    if mode=='factor_verification':
        from dpnn_experiments.factor_verification import run
        run(archive,config,device,radon)
        return
    if mode in ('svd_values_reference','svd_values'):
        from dpnn_experiments.cuda_svd import XGesvd
        if mode=='svd_values_reference':
            with np.load(config['dataset']['small_reference']['path'],allow_pickle=False) as z:
                matrix=z['matrix'].copy();expected=z['singular'].copy()
            np.savez_compressed(archive.path/'values_reference_input.npz',matrix=matrix,expected=expected)
            dense=torch.tensor(matrix,device=device)
        else:
            if preflight['memory_before']['free_bytes']<60*2**30:raise RuntimeError('Insufficient GPU memory')
            dense=MatrixOperator.whitened_radon(radon,device).matrix.to_dense()
            expected=None
        solver=XGesvd(vectors=False);started=time.perf_counter()
        try: _,s,_=solver.factor(dense.T,lambda item:archive.event('actual_workspace',**item))
        finally: solver.close()
        elapsed=time.perf_counter()-started
        save_tensor(archive.path/'singular_values.npy',s)
        result={'mode':mode,'factor_seconds':elapsed,'memory':memory_record(torch),
                'ranks':{str(t):int((s>t*s[0]).sum()) for t in options['cutoffs']},
                'largest_singular_value':float(s[0]),'smallest_singular_value':float(s[-1]),
                'limitations':['Singular values only; no singular vectors or projector']}
        if expected is not None:
            result['relative_error']=float(np.linalg.norm(array(s)-expected)/np.linalg.norm(expected))
            if result['relative_error']>1e-10:raise ArithmeticError('Values-only SVD reference failed')
        write_json(archive.path/'summary.json',result)
        print(json.dumps(result),flush=True)
        return
    if mode in ('block_reference','block_factorization'):
        from dpnn_experiments.block_factorization import run
        if mode=='block_factorization' and preflight['memory_before']['free_bytes'] < options['minimum_free_gpu_GiB']*2**30:
            raise RuntimeError('Insufficient free GPU memory for block factorization')
        run(archive,config,device,radon)
        return
    if mode in ('acceleration_reference','acceleration'):
        from tools import acceleration_diagnostics
        if mode=='acceleration_reference': acceleration_diagnostics.reference(archive,config,device)
        else: acceleration_diagnostics.large(archive,config,device,radon)
        return
    if mode == 'gram_reference':
        from dpnn_experiments.cuda_svd import Xsyevd
        with np.load(config['dataset']['small_reference']['path'],allow_pickle=False) as z:
            matrix=z['matrix'].copy();expected_p=z['null'].copy();expected_s=z['singular'].copy()
        np.savez_compressed(archive.path/'gram_reference_inputs.npz',matrix=matrix,null=expected_p,singular=expected_s)
        dense=torch.tensor(matrix,device=device)
        gram=dense@dense.T
        solver=Xsyevd()
        try: eigenvalues,eigenvectors=solver.factor(gram.T)
        finally: solver.close()
        keep=eigenvalues > 1e-12*eigenvalues[-1]
        w=eigenvectors[:,keep]
        basis=dense.T@w
        p=torch.eye(matrix.shape[1],device=device,dtype=torch.float64)-(basis/eigenvalues[keep])@basis.T
        result={'rank':int(keep.sum()),'sigma_cutoff_relative':1e-6,
                'projector_relative_error':float(np.linalg.norm(array(p)-expected_p)/np.linalg.norm(expected_p)),
                'eigenvalues_relative_error':float(np.linalg.norm(array(eigenvalues)-np.sort(expected_s**2))/np.linalg.norm(expected_s**2))}
        write_json(archive.path/'summary.json',result)
        np.savez_compressed(archive.path/'gram_reference_factors.npz',eigenvalues=array(eigenvalues),eigenvectors=array(eigenvectors),projector=array(p))
        if result['projector_relative_error']>1e-6 or result['eigenvalues_relative_error']>1e-10:
            raise ArithmeticError('64-bit Gram reference failed')
        print(json.dumps(result),flush=True)
        return
    if mode in ('reference','compressed_reference'):
        with np.load(config['dataset']['small_reference']['path'], allow_pickle=False) as z:
            matrix = z['matrix'].copy()
            expected_s = z['singular'].copy()
            expected_p = z['null'].copy()
        np.savez_compressed(archive.path/'small_inputs.npz', matrix=matrix, singular=expected_s, null=expected_p)
        dense = torch.tensor(matrix, device=device)
        if mode=='compressed_reference':
            from dpnn_experiments.cuda_svd import XGesvd
            basis=torch.tensor(np.load(config['dataset']['factor_basis']['path'],allow_pickle=False),device=device)
            reduced=(dense@basis).T.contiguous().T
            solver=XGesvd()
            try: data_u,s,rotation_vh=solver.factor(reduced)
            finally: solver.close()
            u=basis@rotation_vh.T;vh=data_u.T
        elif options['svd_driver']=='Xgesvd':
            from dpnn_experiments.cuda_svd import XGesvd
            solver=XGesvd()
            try: u,s,vh=solver.factor(dense.T.clone(memory_format=torch.preserve_format))
            finally: solver.close()
        else:
            u,s,vh = torch.linalg.svd(dense.T, full_matrices=False, driver='gesvd')
        rank = int((s > options['svd_rtol']*s[0]).sum())
        actual_p = np.eye(matrix.shape[1])-array(u[:,:rank]@u[:,:rank].T)
        compared_s = np.pad(array(s), (0, len(expected_s)-len(s)))
        result = {'rank': rank,
                  'singular_relative_error': float(np.linalg.norm(compared_s-expected_s)/np.linalg.norm(expected_s)),
                  'projector_relative_error': float(np.linalg.norm(actual_p-expected_p)/np.linalg.norm(expected_p)),
                  'reconstruction_relative_error': float(torch.linalg.norm(u@(s[:,None]*vh)-dense.T)/torch.linalg.norm(dense))}
        write_json(archive.path/'summary.json', result)
        np.savez_compressed(archive.path/'small_svd.npz', u=array(u), singular=array(s), vh=array(vh))
        if max(result[k] for k in result if k.endswith('error')) > 1e-8:
            raise ArithmeticError('Direct SVD reference validation failed')
        print(json.dumps(result), flush=True)
        return
    if preflight['memory_before']['free_bytes'] < options['minimum_free_gpu_GiB']*2**30:
        raise RuntimeError('Insufficient free GPU memory for the configured factorization')
    if shutil.disk_usage(archive.path).free < options['minimum_free_disk_GiB']*2**30:
        raise RuntimeError('Insufficient archive space')
    op = MatrixOperator.whitened_radon(radon, device)
    with np.load(config['dataset']['probes']['path'], allow_pickle=False) as z:
        probes = z['probes'].copy()
    np.savez_compressed(archive.path/'inputs.npz', probes=probes)
    values = torch.tensor(probes, device=device)
    torch.cuda.synchronize()
    started = time.perf_counter()
    dense = op.matrix.to_dense()
    torch.cuda.synchronize()
    archive.event('dense_ready', seconds=time.perf_counter()-started, **memory_record(torch))
    print('Dense matrix ready; starting '+mode, flush=True)
    start = time.perf_counter()
    if mode in ('direct_svd','compressed_svd'):
        # M is wide. gesvd requires rows>=columns, so factor M.T.
        if mode=='compressed_svd':
            from dpnn_experiments.cuda_svd import XGesvd
            basis=torch.tensor(np.load(config['dataset']['factor_basis']['path'],allow_pickle=False),device=device)
            reduced=(dense@basis).T.contiguous().T
            archive.event('compressed_matrix_ready',shape=list(reduced.shape),**memory_record(torch))
            del dense
            torch.cuda.empty_cache()
            solver=XGesvd()
            try: data_u,s,rotation_vh=solver.factor(reduced,lambda item:archive.event('actual_workspace',**item))
            finally: solver.close()
            save_tensor(archive.path/'factor_basis_rotation_Vh.npy',rotation_vh)
            u=basis@rotation_vh.T;vh=data_u.T
            del basis,reduced,rotation_vh,data_u
        elif options['svd_driver']=='Xgesvd':
            from dpnn_experiments.cuda_svd import XGesvd
            solver=XGesvd()
            try:
                u,s,vh=solver.factor(dense.T,lambda item:archive.event('actual_workspace',**item))
            finally: solver.close()
        else:
            u,s,vh = torch.linalg.svd(dense.T, full_matrices=False, driver='gesvd')
        torch.cuda.synchronize()
        factor_seconds = time.perf_counter()-start
        archive.event('factorization_finished', seconds=factor_seconds, **memory_record(torch))
        if mode!='compressed_svd':del dense
        torch.cuda.empty_cache()
        # For M.T=U S Vh, M^dagger=U S^dagger Vh.
        save_tensor(archive.path/'factor_image_U.npy', u)
        save_tensor(archive.path/'factor_data_Vh.npy', vh)
        save_tensor(archive.path/'singular_values.npy', s)
        archive.event('factors_saved')
        def make_maps(cutoff):
            rank = int((s > cutoff*s[0]).sum())
            image = u[:,:rank]
            data = vh[:rank,:]
            singular = s[:rank]
            return rank, lambda x:x-image@(image.T@x), lambda y:image@((data@y)/singular)
        reconstruction = u@(s[:,None]*(vh@op.forward(values)))
        original = op.transpose(op.forward(values))
        factor_check = float(torch.linalg.norm(reconstruction-original)/torch.linalg.norm(original))
        # Independent random coefficient orthogonality probes, avoiding a huge U.T U.
        rng = np.random.Generator(np.random.PCG64(config['randomness']['seed']+1))
        c = torch.tensor(rng.normal(size=(u.shape[1],3)), device=device)
        np.savez_compressed(archive.path/'orthogonality_inputs.npz', coefficients=array(c))
        orthogonality = float(torch.linalg.norm(u.T@(u@c)-c)/torch.linalg.norm(c))
        checks = {'transpose_factor_action_relative_error':factor_check,
                  'image_factor_random_orthogonality_error':orthogonality}
        if max(checks.values()) > 1e-8:
            raise ArithmeticError('Large SVD factor validation failed')
    elif mode == 'gram_eigh':
        gram = dense@dense.T
        torch.cuda.synchronize()
        archive.event('gram_ready', seconds=time.perf_counter()-start, **memory_record(torch))
        del dense
        if options.get('gram_driver')=='Xsyevd':
            from dpnn_experiments.cuda_svd import Xsyevd
            solver=Xsyevd()
            try: eigenvalues,eigenvectors=solver.factor(gram.T,lambda item:archive.event('actual_workspace',**item))
            finally: solver.close()
        else:
            eigenvalues, eigenvectors = torch.linalg.eigh(gram, UPLO='L')
        torch.cuda.synchronize()
        factor_seconds = time.perf_counter()-start
        archive.event('factorization_finished', seconds=factor_seconds, **memory_record(torch))
        del gram
        torch.cuda.empty_cache()
        save_tensor(archive.path/'factor_gram_eigenvectors.npy', eigenvectors)
        save_tensor(archive.path/'gram_eigenvalues.npy', eigenvalues)
        archive.event('factors_saved')
        def make_maps(cutoff):
            keep = eigenvalues > cutoff**2*eigenvalues[-1]
            start_index = int((~keep).sum())
            vectors = eigenvectors[:,start_index:]
            eig = eigenvalues[start_index:]
            def inverse(y):
                coefficients = vectors.T@y
                scaled = coefficients/eig if y.ndim==1 else coefficients/eig[:,None]
                return op.transpose(vectors@scaled)
            return len(eig), lambda x:x-inverse(op.forward(x)), inverse
        checks = {'negative_eigenvalues': int((eigenvalues<0).sum()),
                  'minimum_eigenvalue': float(eigenvalues[0]), 'maximum_eigenvalue': float(eigenvalues[-1]),
                  'note':'Normal equations square the singular values. Cutoff sensitivity is part of this diagnostic.'}
    else:
        raise ValueError('Unknown factorization mode')
    records=[]
    for cutoff in options['cutoffs']:
        rank,project,inverse=make_maps(cutoff)
        for j,name in enumerate(config['dataset']['probe_names']):
            v=values[:,j]
            null,times=measure(torch,lambda:project(v),options['timing_repetitions'])
            twice=project(null)
            inverse_value=inverse(op.forward(v))
            q=v-null
            record={'cutoff_relative_sigma':cutoff,'numerical_rank':rank,'probe':name,
                    'idempotence_defect_norm':float(torch.linalg.norm(twice-null)),
                    'inverse_of_matched_data_vs_range_absolute':float(torch.linalg.norm(inverse_value-q)),
                    'orthogonality_inner_product':float(torch.dot(null,q)),
                    'projection_seconds':times, **dp_metrics(op,v,null,options['beta_fractions'])}
            records.append(record)
            np.savez_compressed(archive.path/f'projection_{cutoff:.0e}_{name}.npz',null=array(null),q=array(q),
                                twice=array(twice),inverse_matched=array(inverse_value))
            write_json(archive.path/'projection_results.json',records)
            archive.event('projection_checked',**record)
    # Fixed-factor derivative: no differentiation of SVD/eigh or rank selection.
    rank,project,inverse=make_maps(options['gradient_cutoff'])
    rng=np.random.Generator(np.random.PCG64(config['randomness']['seed']+2))
    direction=torch.tensor(rng.normal(size=n),device=device); direction/=torch.linalg.norm(direction)
    upstream=torch.tensor(rng.normal(size=n),device=device); upstream/=torch.linalg.norm(upstream)
    gradients=[directional_check(archive,config,project,values[:,0],upstream,direction,'factor_null')]
    data=torch.tensor(rng.normal(size=m),device=device); data/=torch.linalg.norm(data)
    data_direction=torch.tensor(rng.normal(size=m),device=device); data_direction/=torch.linalg.norm(data_direction)
    gradients.append(directional_check(archive,config,inverse,data,upstream,data_direction,'factor_inverse'))
    summary={'mode':mode,'factor_seconds':factor_seconds,'factor_checks':checks,'projections':records,
             'gradients':gradients,'memory':memory_record(torch),
             'limitations':['Explicit numerical spectral cutoffs; not proof of exact algebraic rank',
                            'Two operator-development inputs; no network training',
                            'Factors are fixed; no derivative through factorization or cutoff']}
    write_json(archive.path/'summary.json',summary)
    print(json.dumps(summary),flush=True)


def supervise(archive, config, root, worker_module='tools.factorization_diagnostics'):
    stop=threading.Event()
    def monitor():
        while not stop.is_set():
            try:
                sample=resources()
            except Exception as exc:
                sample={'time_utc':utc_now(),'error':repr(exc)}
            with (archive.path/'resources.jsonl').open('a') as f:
                f.write(json.dumps(sample)+'\n');f.flush();os.fsync(f.fileno())
            stop.wait(config['computation']['resource_sample_seconds'])
    thread=threading.Thread(target=monitor,daemon=True)
    thread.start()
    process=subprocess.Popen([sys.executable,'-u','-m',worker_module,
                              '--worker',str(archive.path),'--root',str(root)],
                             stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    def output():
        for line in process.stdout:
            print(line,end='',flush=True)
    reader=threading.Thread(target=output,daemon=True);reader.start()
    try:
        try:
            code=process.wait(timeout=config['computation']['timeout_seconds'])
        except subprocess.TimeoutExpired:
            process.terminate()
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired: process.kill();process.wait()
            archive.event('worker_timeout',returncode=process.returncode)
            raise TimeoutError('Configured factorization wall-time budget exhausted')
        if code:
            raise RuntimeError(f'Numerical worker exited with status {code}')
    finally:
        reader.join(timeout=15)
        stop.set();thread.join(timeout=15)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--worker',type=Path)
    args=parser.parse_args()
    if args.worker:
        manifest=json.loads((args.worker/'manifest.json').read_text())
        run_worker(WorkerArchive(args.worker),manifest['configuration'],args.root)
        return
    config=json.loads(args.config.read_text())
    config['dataset']['reference_operator']=reference_metadata(args.root/'reference')
    for key in ('small_reference','probes'):
        relative=config['dataset'][key]
        path=args.root/relative
        config['dataset'][key]={'path':str(path),'sha256':digest(path)}
    if 'factor_basis' in config['dataset']:
        path=args.root/config['dataset']['factor_basis']
        completion=json.loads((path.parent/'completion.json').read_text())
        if completion['status']!='completed':raise ValueError('Factorization did not complete successfully')
        indexed={item['path']:item for item in json.loads((path.parent/'artifacts.json').read_text())['files']}
        actual=digest(path)
        if actual!=indexed[path.name]['sha256']:raise ValueError('Basis hash differs from canonical artifact index')
        config['dataset']['factor_basis']={'path':str(path),'sha256':actual,'run_id':path.parent.name}
    for item in config['dataset'].get('comparison_outputs',[]):
        path=args.root/item['path'];item['path']=str(path);item['sha256']=digest(path)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=config['computation']['cublas_workspace_config']
    with RunArchive(config,args.root/'runs',SOURCE_ROOT,args.root/'metadata',
                    parent_run_id=config['dataset']['reference_operator']['run_id']) as archive:
        print('Run archive: '+str(archive.path),flush=True)
        for key in ('small_reference','probes'):
            reference=config['dataset'][key]
            if digest(reference['path'])!=reference['sha256']:
                raise ValueError('Input changed after manifest creation')
        ref_dir=archive.path/'reference';ref_dir.mkdir()
        for name in ('manifest.json','completion.json','artifacts.json','operator.npz'):
            shutil.copyfile(args.root/'reference'/name,ref_dir/name)
        supervise(archive,config,args.root)


if __name__=='__main__':
    main()
