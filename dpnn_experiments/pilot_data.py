"""Independent analytic ellipse data and checked Tikhonov starts for a pilot."""
import argparse
import copy
import json
import math
import os
import shutil
from pathlib import Path
import time

import numpy as np
import torch

from dpnn_experiments.gpu_benchmark import reference_metadata, load_operator
from dpnn_experiments.operators import Geometry
from dpnn_experiments.phantoms import rng_for, pixel_means, analytic_sinogram
from dpnn_experiments.projection_diagnostics import setup, array
from dpnn_experiments.projections import MatrixOperator
from dpnn_experiments.provenance import digest, write_json
from dpnn_experiments.training_archive import TrainingArchive, file_reference

SOURCE_ROOT=Path(__file__).resolve().parents[1]


def tikhonov(operator,y,alpha,rtol,atol,max_iterations):
    """Batched CG for (M.T M + alpha I)x~=M.T y~, with actual residual check."""
    if alpha<=0 or rtol<=0 or atol<=0 or max_iterations<1:raise ValueError('Invalid CG configuration')
    b=operator.transpose(y);x=torch.zeros_like(b);r=b.clone();p=r.clone()
    rr=(r*r).sum(dim=0);threshold=atol+rtol*torch.linalg.vector_norm(b,dim=0)
    done=torch.sqrt(rr)<=threshold;iterations=torch.zeros(y.shape[1],dtype=torch.int64,device=y.device)
    history=[]
    for i in range(max_iterations):
        if bool(done.all()):break
        hp=operator.transpose(operator.forward(p))+alpha*p
        denominator=(p*hp).sum(dim=0)
        if bool(((denominator<=0)&~done).any()):raise ArithmeticError('CG lost positive definiteness')
        step=torch.where(done,torch.zeros_like(rr),rr/torch.where(done,torch.ones_like(rr),denominator))
        x=x+p*step;r=r-hp*step
        new_rr=(r*r).sum(dim=0)
        iterations=torch.where(done,iterations,torch.full_like(iterations,i+1))
        done=done|(torch.sqrt(new_rr)<=threshold)
        ratio=torch.where(done,torch.zeros_like(rr),new_rr/torch.where(rr>0,rr,torch.ones_like(rr)))
        p=torch.where(done[None,:],torch.zeros_like(p),r+p*ratio)
        rr=new_rr
        history.append({'iteration':i+1,'residual_estimates':array(torch.sqrt(rr)).tolist()})
    actual=torch.linalg.vector_norm(b-operator.transpose(operator.forward(x))-alpha*x,dim=0)
    info={'iterations':array(iterations).tolist(),'actual_normal_residual':array(actual).tolist(),
          'acceptance_threshold':array(threshold).tolist(),'history':history,
          'passed':bool((actual<=threshold).all()),
          'meaning':'Normal residual of the regularized system; no claim of exact inverse'}
    return x,info


def ellipse_specimen(config,object_id):
    p=config['dataset']['generator'];g=rng_for(config['randomness']['geometry_seed'],object_id,'geometry')
    n=int(g.integers(p['count_min'],p['count_max']+1))
    amplitudes=g.dirichlet(np.full(n,p['dirichlet_alpha']))*g.uniform(*p['total_amplitude_range'])
    ellipses=[];attempts=[]
    for amplitude in amplitudes:
        for attempt in range(p['max_rejection_attempts']):
            major=g.uniform(*p['major_axis_range']);minor=major*g.uniform(*p['axis_ratio_range'])
            center=g.uniform(-p['support_radius'],p['support_radius'],2)
            if np.linalg.norm(center)+major<=p['support_radius']:break
        else:raise ValueError('Ellipse rejection limit')
        attempts.append(attempt+1)
        ellipses.append({'center':center.tolist(),'axes':[float(major),float(minor)],
                         'rotation_deg':float(g.uniform(0,180)),'amplitude':float(amplitude)})
    return {'id':object_id,'family':'nonnegative_random_ellipses','ellipses':ellipses,
            'rejection_attempts':attempts}


def prepare(config,root,operator_root):
    root=Path(root);config=copy.deepcopy(config)
    ref=reference_metadata(operator_root)
    if ref['geometry']!=config['geometry'] or ref['files']['operator.npz']!=config['operator_sha256']:
        raise ValueError('Pilot operator differs from pinned geometry/hash')
    splits=config['dataset']['split_ids'];flat=[x for ids in splits.values() for x in ids]
    if set(splits)!={'train','validation','test'} or any(not v for v in splits.values()) or len(flat)!=len(set(flat)):
        raise ValueError('Require nonempty disjoint train/validation/test object IDs')
    config['dataset']['reference_operator']=ref
    config['training']={'applicable':False,'reason':'Synthetic data and cached regularized starts only'}
    os.environ['CUBLAS_WORKSPACE_CONFIG']=config['computation']['cublas_workspace_config']
    with TrainingArchive(config,root/'runs',SOURCE_ROOT,root/'metadata') as archive:
        print('Data archive: '+str(archive.path),flush=True)
        device=setup(archive,config);radon=load_operator(operator_root,ref)
        operator=MatrixOperator.whitened_radon(radon,device);g=radon.geometry
        specimens=[ellipse_specimen(config,name) for name in flat]
        write_json(archive.path/'specimens.json',specimens)
        write_json(archive.path/'splits.json',splits);archive.flush_metadata()
        records=[];batch=config['computation']['preparation_batch_size'];sqrtwx=2/g.size
        # A committed batch is the recovery unit. Never accept unsealed files or
        # samples from a different geometry/noise/solver contract.
        previous={}
        if config.get('reuse_completions'):
            from dpnn_experiments.main_integrity import SealedArchive, read
            from dpnn_experiments.main_inputs import normalized
            for completion in config['reuse_completions']:
                from dpnn_experiments.training_archive import verify_reference
                old=SealedArchive(verify_reference(completion).parent,statuses=('failed','interrupted'))
                if normalized(old.configuration)!=normalized(config):raise ValueError('Data recovery contract differs')
                for member in sorted(k for k in old.entries if k.startswith('batches/') and k.endswith('.json')):
                    item=read(old.member(member));offset=item['first']
                    if offset in previous:raise ValueError('Duplicate recovery batch')
                    trace=read(old.reference(item['solver']))
                    if not trace['passed']:raise ValueError('Unqualified data recovery batch')
                    expected=flat[offset:offset+batch]
                    if [r['id'] for r in item['records']]!=expected:raise ValueError('Recovery batch IDs differ')
                    for rec in item['records']:old.reference(rec['file'])
                    previous[offset]=item
        for start in range(0,len(flat),batch):
            if start in previous:
                item=previous[start];copied=[]
                trace_path=archive.path/f'cg_batch_{start:05d}.json'
                shutil.copyfile(item['solver']['path'],trace_path)
                for j,record in enumerate(item['records']):
                    path=archive.path/'samples'/f'{start+j:05d}.npz';path.parent.mkdir(exist_ok=True)
                    shutil.copyfile(record['file']['path'],path)
                    copied.append({**record,'file':file_reference(path)})
                records.extend(copied)
                write_json(archive.path/f'batches/{start:05d}.json',{'first':start,'records':copied,'solver':file_reference(trace_path),'reused':True})
                archive.event('data_batch_reused',first=start,count=len(copied));continue
            chosen=specimens[start:start+batch];images=[];clean=[];noisy=[];deltas=[]
            for specimen in chosen:
                image=pixel_means(specimen,g.size,config['computation']['pixel_quadrature'])
                yy=analytic_sinogram(specimen,g.theta,g.positions).reshape(-1)*math.sqrt(g.data_weight)
                delta=config['computation']['relative_noise_norm']*np.linalg.norm(yy)
                generator=rng_for(config['randomness']['noise_seed'],specimen['id'],'noise-000')
                direction=generator.normal(size=len(yy));direction/=np.linalg.norm(direction)
                images.append(image);clean.append(yy);noisy.append(yy+delta*direction);deltas.append(delta)
            y=torch.tensor(np.stack(noisy).T,device=device,dtype=torch.float64)
            settings=config['computation']['tikhonov'];began=time.perf_counter()
            with torch.no_grad():
                solution,trace=tikhonov(operator,y,**settings)
                starts=array(solution/sqrtwx).T.reshape(-1,1,g.size,g.size)
                raster_y=array(operator.forward(torch.tensor(np.stack(images).reshape(len(chosen),-1).T*sqrtwx,device=device)))
            trace['wall_seconds']=time.perf_counter()-began
            write_json(archive.path/f'cg_batch_{start:05d}.json',trace)
            if not trace['passed']:raise ArithmeticError('Tikhonov normal residual tolerance not met')
            for j,specimen in enumerate(chosen):
                idx=start+j;name=flat[idx];split=next(k for k,ids in splits.items() if name in ids)
                target=np.asarray(images[j])[None,:,:];path=archive.path/'samples'/f'{idx:05d}.npz'
                path.parent.mkdir(exist_ok=True)
                beta=config['computation']['beta_noise_multiplier']*deltas[j]
                np.savez_compressed(path,target=target,start=starts[j],measurement=np.asarray(noisy[j]),
                    analytic_clean=np.asarray(clean[j]),raster_clean=raster_y[:,j],
                    delta=deltas[j],beta=beta)
                model_error=float(np.linalg.norm(raster_y[:,j]-clean[j]))
                records.append({'id':name,'split':split,'family':specimen['family'],'file':file_reference(path),
                    'noise_realization_id':name+'/noise-000','delta':float(deltas[j]),'beta':float(beta),
                    'model_error_norm':model_error,
                    'combined_noise_model_error_norm':float(np.linalg.norm(raster_y[:,j]-noisy[j])),
                    'realized_noise_norm':float(np.linalg.norm(noisy[j]-clean[j])),
                    'alpha':settings['alpha'],'cg_iterations':trace['iterations'][j],
                    'cg_normal_residual':trace['actual_normal_residual'][j]})
            archive.event('data_batch_saved',first=start,count=len(chosen),solver=trace['actual_normal_residual'])
            write_json(archive.path/f'batches/{start:05d}.json',{'first':start,'records':records[-len(chosen):],
                'solver':file_reference(archive.path/f'cg_batch_{start:05d}.json'),'reused':False})
            archive.flush_metadata()
            print(f'Prepared {len(records)}/{len(flat)} phantoms',flush=True)
            if config.get('stop_after_batches') and (start//batch+1)>=config['stop_after_batches']:
                raise InterruptedError('Requested technical data batch stop')
        dataset={'schema':1,'source_run':archive.run_id,'geometry':config['geometry'],
                 'operator_sha256':config['operator_sha256'],'coordinates':'physical images; whitened weighted measurements',
                 'split_ids':splits,'records':records,'configuration':config,
                 'test_policy':'Generated and reserved; never read by the training/selection loader'}
        write_json(archive.path/'dataset.json',dataset)
        write_json(archive.path/'summary.json',{'objects':len(records),'counts':{k:len(v) for k,v in splits.items()},
            'max_normal_residual':max(x['cg_normal_residual'] for x in records),
            'max_noise_norm_error':max(abs(x['delta']-x['realized_noise_norm']) for x in records),
            'mean_model_error_norm':float(np.mean([x['model_error_norm'] for x in records]))})
    return archive.path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--operator-root',type=Path,required=True)
    a=p.parse_args();prepare(json.loads(a.config.read_text()),a.root,a.operator_root)


if __name__=='__main__':main()
