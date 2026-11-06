"""Durable training segments, immutable manifests and linked atomic checkpoints.

Every evaluated validation state has a preserved weight file; training steps
use either full retention or explicit run/step lineage. Optimizer checkpoints refer
to that file by hash instead of duplicating it. Resume always opens a new run.
No remote tracking service is required.
"""
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback
import uuid

import numpy as np
import torch

from dpnn_experiments.provenance import RunArchive, digest, utc_now, write_json


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def append_json(path, row):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    encoded=json.dumps(row,sort_keys=True,allow_nan=False)+'\n'
    with path.open('a',encoding='utf-8') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        f.write(encoded);f.flush();os.fsync(f.fileno())
        fcntl.flock(f,fcntl.LOCK_UN)


def atomic_torch_save(value,path,writer=None):
    """Never replace an existing immutable artifact; remove incomplete temporaries."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with tmp.open('xb') as f:
            (writer or torch.save)(value,f)
            f.flush();os.fsync(f.fileno())
        # Link gives atomic publication without overwriting an existing filename.
        os.link(tmp,path)
        fd=os.open(path.parent,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
    finally:
        tmp.unlink(missing_ok=True)
    return {'path':str(path),'sha256':digest(path),'bytes':path.stat().st_size}


def file_reference(path):
    p=Path(path).resolve()
    return {'path':str(p),'sha256':digest(p),'bytes':p.stat().st_size}


def verify_reference(ref):
    if set(ref) != {'path','sha256','bytes'}:raise ValueError('Incomplete artifact reference')
    p=Path(ref['path'])
    if not p.is_file() or p.stat().st_size!=ref['bytes'] or digest(p)!=ref['sha256']:
        raise ValueError('Artifact integrity failure: '+str(p))
    return p


def rng_state(sampler,device='cpu'):
    device=torch.device(device)
    n=np.random.get_state()
    return {'python':random.getstate(),
            'numpy':{'name':n[0],'keys':torch.from_numpy(n[1].astype(np.int64)),
                     'position':n[2],'has_gauss':n[3],'cached_gaussian':n[4]},
            'torch_cpu':torch.get_rng_state(),
            'torch_cuda':([torch.cuda.get_rng_state(device)] if device.type=='cuda' else []),
            'cuda_scope':'training_device_only',
            'sampler':sampler.get_state()}


def restore_rng(state,sampler,device='cpu'):
    device=torch.device(device)
    if bool(state['torch_cuda'])!=(device.type=='cuda'):
        raise ValueError('Checkpoint RNG device mismatch')
    if state['torch_cuda'] and (state.get('cuda_scope')!='training_device_only' or len(state['torch_cuda'])!=1):
        raise ValueError('Checkpoint must bind RNG to the single training device')
    random.setstate(state['python']);n=state['numpy']
    np.random.set_state((n['name'],n['keys'].cpu().numpy().astype(np.uint32),
                         n['position'],n['has_gauss'],n['cached_gaussian']))
    torch.set_rng_state(state['torch_cpu'].cpu())
    if state['torch_cuda']:torch.cuda.set_rng_state(state['torch_cuda'][0].cpu(),device)
    sampler.set_state(state['sampler'].cpu())


class TrainingArchive(RunArchive):
    """Manifest is immutable after entry; dynamic results live in separate files."""
    def __enter__(self):
        super().__enter__()
        self.manifest_hash=digest(self.path/'manifest.json')
        self.outcome='completed';self.reason=None;self.last_checkpoint=None
        self.index_path=self.path.parent.parent/'training_index.jsonl'
        self.index('running')
        return self

    def index(self,status):
        append_json(self.index_path,{'run_id':self.run_id,'time_utc':utc_now(),'status':status,
                    'path':str(self.path),'parent_run_id':self.parent_run_id,
                    'manifest_sha256':self.manifest_hash,'last_checkpoint':self.last_checkpoint})

    def rows(self,name,records):
        for record in records:append_json(self.path/(name+'.jsonl'),record)

    def flush_metadata(self):
        if digest(self.path/'manifest.json')!=self.manifest_hash:
            raise RuntimeError('Immutable training manifest was changed')
        self._mirror_metadata()

    def weights(self,model,step):
        p=self.path/'weights'/f'step_{step:08d}.pt'
        if p.exists():return file_reference(p)
        ref=atomic_torch_save(model.state_dict(),p)
        self.rows('weights',[{'step':step,**ref}])
        return ref

    def step_weights(self,model,step,retain):
        if retain or (self.path/'weights'/f'step_{step:08d}.pt').exists():
            return self.weights(model,step)
        return {'run_id':self.run_id,'step':step,'retained':False,
                'recovery':'last committed checkpoint; subsequent steps may require replay'}

    def checkpoint(self,model,optimizer,scheduler,sampler,progress,binding,label):
        weights=self.weights(model,progress['step'])
        state={'schema':1,'binding':binding,'weight_reference':weights,
               'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
               'optimizer_runtime':{'step_called':bool(getattr(optimizer,'_opt_called',False))},
               'rng':rng_state(sampler,model.projection.basis.device),'progress':copy.deepcopy(progress),
               'mixed_precision_scaler':None,'run_id':self.run_id,
               'manifest_sha256':self.manifest_hash,'created_utc':utc_now()}
        ref=atomic_torch_save(state,self.path/'checkpoints'/f"step_{progress['step']:08d}_{label}.pt")
        self.rows('checkpoints',[{'label':label,'progress':copy.deepcopy(progress),
                                'weight_reference':weights,**ref}])
        self.last_checkpoint=ref
        write_json(self.path/'latest_checkpoint.json',ref)
        self.event('checkpoint_committed',label=label,step=progress['step'],checkpoint=ref,
                   weight_reference=weights)
        self.flush_metadata();self.index('running')
        return ref

    def __exit__(self,exc_type,exc,tb):
        status=self.outcome if exc is None else 'failed'
        try:
            if exc_type:traceback.print_exception(exc_type,exc,tb)
            self.event('finished',status=status,reason=self.reason,error=repr(exc) if exc else None,
                       elapsed_seconds=time.perf_counter()-self.start)
        except BaseException:
            status='incomplete'
            raise
        finally:
            self._stack.close();self.stdout.close();self.stderr.close()
            try:
                if digest(self.path/'manifest.json')!=self.manifest_hash:
                    raise RuntimeError('Immutable training manifest changed')
                write_json(self.path/'outcome.json',{'status':status,'reason':self.reason,
                    'error':repr(exc) if exc else None,'last_checkpoint':self.last_checkpoint,
                    'time_utc':utc_now(),'elapsed_seconds':time.perf_counter()-self.start})
                entries=[{'path':str(p.relative_to(self.path)),'bytes':p.stat().st_size,'sha256':digest(p)}
                         for p in sorted(self.path.rglob('*')) if p.is_file()
                         and p.name not in ('artifacts.json','completion.json')]
                write_json(self.path/'artifacts.json',{'files':entries})
                write_json(self.path/'completion.json',{'status':status,'error':repr(exc) if exc else None,
                    'time_utc':utc_now(),'manifest_sha256':self.manifest_hash,
                    'artifacts_sha256':digest(self.path/'artifacts.json')})
                self.flush_metadata();self.index(status)
            except BaseException as error:
                write_json(self.path/'completion.json',{'status':'incomplete','error':repr(error),
                                                       'time_utc':utc_now()})
                raise
        return False


def audit_runs(root):
    """A missing completion marker is never inferred to mean success."""
    records=[]
    for p in sorted((Path(root)/'runs').glob('*')):
        if not (p/'manifest.json').exists():continue
        completion=json.loads((p/'completion.json').read_text()) if (p/'completion.json').exists() else None
        status=completion['status'] if completion else 'unfinished_or_running'
        verified=0
        if completion and 'artifacts_sha256' in completion:
            if digest(p/'artifacts.json')!=completion['artifacts_sha256']:raise ValueError('Index altered')
            for item in json.loads((p/'artifacts.json').read_text())['files']:
                verify_reference({'path':str(p/item['path']),'bytes':item['bytes'],'sha256':item['sha256']})
                verified+=1
        records.append({'run_id':p.name,'status':status,'artifact_entries_verified':verified})
    return records
