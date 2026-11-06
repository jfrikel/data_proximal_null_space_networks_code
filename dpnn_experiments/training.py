"""Single-device Float64 training/evaluation with complete artifacts and resume.

Supported profile: fixed cached noise, no augmentation, synchronous batches,
Adam, StepLR, MSE, RES/NSN/shared scalar DP. Unsupported modes fail explicitly.
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import resource
import signal
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from dpnn_experiments.network_layers import FixedBlockProjection, NullSpaceNetwork, ScalarDataProximalNetwork, SeparateDataProximalNetwork
from dpnn_experiments.projection_diagnostics import setup, array
from dpnn_experiments.provenance import digest, write_json
from dpnn_experiments.training_archive import TrainingArchive, canonical_hash, file_reference, verify_reference, restore_rng, audit_runs

SOURCE_ROOT=Path(__file__).resolve().parents[1]


def validate_config(c):
    required={'schema_version','study_id','purpose','mode','dataset','geometry','randomness','computation',
              'model','optimizer','scheduler','training','selection','logging','metrics','execution','projection'}
    if required-set(c):raise ValueError('Missing configuration fields: '+str(sorted(required-set(c))))
    sections={
      'dataset':{'reference','used_ids','normalization','augmentation','loader'},
      'randomness':{'seed','model_seed','sampler_seed','python_seed','numpy_seed','generator'},
      'computation':{'device','dtype','cpu_threads','deterministic_algorithms','tf32','cublas_workspace_config'},
      'model':{'architecture','method','in_channels','out_channels','bilinear','width','dropout','initialization'},
      'optimizer':{'name','lr','betas','eps','weight_decay','amsgrad','foreach','fused','maximize','capturable','differentiable'},
      'scheduler':{'name','step_size','gamma','last_epoch'},
      'training':{'applicable','epochs','batch_size','loss','gradient_accumulation','gradient_clip_norm','mixed_precision'},
      'selection':{'split','metric','direction','patience','min_delta'},
      'logging':{'checkpoint_every_steps','weights_every_step','save_predictions','mirror_every_checkpoint'},
      'metrics':{'data_range','ssim_window','ssim_sigma','ssim_k1','ssim_k2','ssim_covariance','ssim_padding','proximity_atol','aggregation'},
      'execution':{'resume','stop_after_step','injected_io_failure_step'},
    }
    if 'early_stopping_enabled' in c['selection']:
        sections['selection'].add('early_stopping_enabled')
        if type(c['selection']['early_stopping_enabled']) is not bool:
            raise ValueError('early_stopping_enabled must be a boolean')
    if 'continuation' in c['execution']:
        sections['execution'].add('continuation')
    for section,keys in sections.items():
        if set(c[section])!=keys:raise ValueError(f'Incomplete or unknown {section} fields: {set(c[section])^keys}')
    if c['mode'] not in ('train','evaluate'):raise ValueError('Unknown execution mode')
    if c['computation']['dtype']!='float64' or c['computation']['tf32'] or not c['computation']['deterministic_algorithms']:
        raise ValueError('Only deterministic Float64/TF32-off supported')
    if c['model']['architecture'] not in ('unet','tiny_dropout') or c['model']['method'] not in ('RES','NSN','DP','DP_TWO'):
        raise ValueError('Unknown architecture/method')
    if c['model']['in_channels']!=1 or c['model']['out_channels']!=1 or not c['model']['bilinear']:
        raise ValueError('Unsupported channel/upsampling profile')
    if c['model']['initialization']!='torch_default_seeded':raise ValueError('Unsupported initialization')
    if c['model']['architecture']=='unet' and (c['model']['width'] is not None or c['model']['dropout']!=0.):
        raise ValueError('Existing UNet has fixed width and no dropout')
    if c['optimizer']['name']!='Adam' or c['scheduler']['name']!='StepLR':raise ValueError('Unsupported optimizer/scheduler')
    if any(c['optimizer'][x] for x in ('foreach','fused','maximize','capturable','differentiable')):
        raise ValueError('Unsupported Adam execution mode')
    if c['scheduler']['last_epoch']!=-1:raise ValueError('Initial last_epoch must be -1; resume restores state')
    if c['training']['loss']!='mean_pixel_MSE' or c['training']['gradient_accumulation']!=1 or c['training']['mixed_precision']:
        raise ValueError('Unsupported loss/accumulation/precision')
    for section,key in [('training','epochs'),('training','batch_size'),('logging','checkpoint_every_steps'),('selection','patience')]:
        if type(c[section][key]) is not int or c[section][key]<1:raise ValueError('Require positive '+key)
    if c['selection']['split']!='validation' or c['selection']['metric']!='mse' or c['selection']['direction']!='min':
        raise ValueError('Selection must use validation MSE only')
    if c['dataset']['normalization']!='none_physical_intensity' or c['dataset']['augmentation']!='none':
        raise ValueError('Only recorded physical intensities/no augmentation supported')
    if c['dataset']['loader']!={'num_workers':0,'shuffle':True,'drop_last':False,'pin_memory':False,'prefetch':0}:
        raise ValueError('Only synchronous complete batches supported')
    if type(c['logging']['weights_every_step']) is not bool or not c['logging']['save_predictions'] or not c['logging']['mirror_every_checkpoint']:
        raise ValueError('Mandatory artifact retention disabled')
    if c['metrics']['ssim_covariance']!='population' or c['metrics']['ssim_padding']!='valid':
        raise ValueError('Unsupported SSIM convention')
    if c['metrics']['ssim_window']%2!=1 or not 1<=c['metrics']['ssim_window']<=c['geometry']['size']:
        raise ValueError('SSIM window does not fit image')
    if c['metrics']['aggregation']!='per_object_mean_median_population_std_min_max_no_CI':
        raise ValueError('Unsupported aggregation')
    json.dumps(c,allow_nan=False)


class CachedData:
    def __init__(self,config):
        ref=config['dataset']['reference'];p=verify_reference(ref)
        self.manifest=json.loads(p.read_text());self.records={x['id']:x for x in self.manifest['records']}
        splits=self.manifest['split_ids'];flat=[x for ids in splits.values() for x in ids]
        if len(flat)!=len(set(flat)) or len(flat)!=len(self.records):raise ValueError('Overlapping/duplicate dataset IDs')
        if self.manifest['geometry']!=config['geometry']:raise ValueError('Dataset geometry mismatch')
        if config['projection']['kind']=='block' and self.manifest.get('operator_sha256')!=config['projection']['operator_sha256']:
            raise ValueError('Dataset and layer operator hashes differ')
        self.used=config['dataset']['used_ids']
        if set(self.used)!={'train','validation'} or any(not x for x in self.used.values()):
            raise ValueError('Require train/validation subsets; no test loader in training')
        for split,ids in self.used.items():
            if len(ids)!=len(set(ids)) or not set(ids)<=set(splits[split]):raise ValueError('Invalid subset')
            for name in ids:
                if self.records[name]['split']!=split:raise ValueError('Split metadata mismatch')
                verify_reference(self.records[name]['file'])
        self.read_ids=[]

    def batch(self,ids,device,*,split):
        if split not in ('train','validation') or not set(ids)<=set(self.used[split]):
            raise ValueError('Attempt to read outside declared training/validation subset')
        values=[]
        for name in ids:
            with np.load(self.records[name]['file']['path'],allow_pickle=False) as d:
                values.append({k:d[k].copy() for k in ('start','target','measurement','delta','beta')})
            self.read_ids.append(name)
        return {k:torch.tensor(np.stack([v[k] for v in values]),device=device,dtype=torch.float64) for k in values[0]}


class TinyCorrection(nn.Module):
    def __init__(self,width,dropout):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(1,width,3,padding=1),nn.Tanh(),nn.Dropout(dropout),nn.Conv2d(width,1,1))
    def forward(self,z):return self.net(z)


class ResidualNetwork(NullSpaceNetwork):
    def forward(self,z,*,return_details=False):
        value=self.correction(z);out=z+value
        if return_details:return out, {'value':value,'control':1.}
        return out


def build_model(config,device,archive):
    p=config['projection']
    if p['kind']=='block':
        projection=FixedBlockProjection.from_archive(p['factor_run'],basis_sha256=p['basis_sha256'],
            operator_sha256=p['operator_sha256'],geometry=config['geometry'],verification_run=p['verification_run'],
            verification_summary_sha256=p['verification_summary_sha256'],coordinates='physical',device=device)
    elif p['kind']=='fixture':
        path=verify_reference(p['reference'])
        with np.load(path,allow_pickle=False) as d:
            projection=FixedBlockProjection(torch.tensor(d['basis'],device=device),torch.tensor(d['matrix'],device=device),
                coordinates='physical',image_weight=float(d['image_weight']),
                contract={'kind':'logger_fixture','sha256':p['reference']['sha256'],'geometry':config['geometry']})
    else:raise ValueError('Unknown projection kind')
    model=build_correction_model(config,projection,device)
    write_json(archive.path/'projection_contract.json',projection.contract)
    archive.event('model_ready',parameters=sum(p.numel() for p in model.parameters()),contract=projection.contract)
    return model


def build_correction_model(config,projection,device):
    def make():
        if config['model']['architecture']=='unet':
            from src.unet import UNet
            correction=UNet(1,1,bilinear=True)
        else:correction=TinyCorrection(config['model']['width'],config['model']['dropout'])
        return correction.double().to(device)
    if config['model']['method']=='DP_TWO':
        return SeparateDataProximalNetwork(make(),make(),projection)
    klass={'RES':ResidualNetwork,'NSN':NullSpaceNetwork,'DP':ScalarDataProximalNetwork}[config['model']['method']]
    return klass(make(),projection)


def forward(model,batch,method,details=False):
    if method in ('DP','DP_TWO'):return model(batch['start'],batch['beta'],return_details=details)
    output=model(batch['start'])
    return (output,None) if details else output


def ssim(pred,target,c):
    n=c['ssim_window'];v=torch.arange(n,dtype=pred.dtype,device=pred.device)-(n-1)/2
    g=torch.exp(-v.square()/(2*c['ssim_sigma']**2));g/=g.sum();kernel=(g[:,None]*g[None,:])[None,None]
    mx=F.conv2d(pred,kernel);my=F.conv2d(target,kernel)
    vx=F.conv2d(pred.square(),kernel)-mx.square();vy=F.conv2d(target.square(),kernel)-my.square()
    cov=F.conv2d(pred*target,kernel)-mx*my
    c1=(c['ssim_k1']*c['data_range'])**2;c2=(c['ssim_k2']*c['data_range'])**2
    return (((2*mx*my+c1)*(2*cov+c2))/((mx.square()+my.square()+c1)*(vx+vy+c2))).flatten(1).mean(1)


def aggregate(rows):
    keys=('mse','relative_l2','psnr','ssim','data_change','data_residual','null_leak','signed_beta_excess')
    result={}
    for k in keys:
        a=np.array([r[k] for r in rows if r.get(k) is not None],dtype=float)
        result[k]={'count':len(a),'mean':float(a.mean()),'median':float(np.median(a)),
                   'population_std':float(a.std()),'min':float(a.min()),'max':float(a.max())} if len(a) else {'count':0}
    return result


def evaluate(archive,config,model,data,weight_reference,step,label,baseline=False):
    model.eval();rows=[];ids=data.used['validation'];method=config['model']['method']
    event_id=f'{archive.run_id}/{label}/step-{step:08d}'
    archive.rows('evaluations',[{'evaluation_id':event_id,'split':'validation','ids':ids,
        'weight_reference':None if baseline else weight_reference,'step':step,'baseline':baseline,'status':'started'}])
    predictions=[]
    with torch.no_grad():
        for start in range(0,len(ids),config['training']['batch_size']):
            names=ids[start:start+config['training']['batch_size']]
            b=data.batch(names,model.projection.basis.device,split='validation')
            out=b['start'] if baseline else forward(model,b,method)
            diff=out-b['target'];mse=diff.square().flatten(1).mean(1)
            target_norm=b['target'].flatten(1).norm(dim=1)
            error_norm=diff.flatten(1).norm(dim=1)
            similarity=ssim(out,b['target'],config['metrics'])
            change=model.projection.data(out-b['start']).norm(dim=1)
            residual=(model.projection.data(out)-b['measurement']).norm(dim=1)
            if baseline:
                leak=torch.zeros_like(change);budget=torch.zeros_like(change);control=torch.zeros_like(change)
            else:
                if method=='DP_TWO':
                    _,details=forward(model,b,method,True);null=details['null']
                else:
                    value=model.correction(b['start']);null=model.projection(value)
                leak=model.projection.data(null).norm(dim=1)
                budget=b['beta'] if method in ('DP','DP_TWO') else torch.zeros_like(change)
                if method in ('DP','DP_TWO'):_,d=forward(model,b,method,True);control=d['control']
                else:control=torch.zeros_like(change)
            for j,name in enumerate(names):
                constrained=baseline or method in ('NSN','DP','DP_TWO')
                row={'evaluation_id':event_id,'id':name,'noise_realization_id':data.records[name]['noise_realization_id'],
                    'weight_sha256':None if baseline else weight_reference['sha256'],'step':step,
                    'mse':float(mse[j]),
                    'relative_l2':float(error_norm[j]/target_norm[j]) if target_norm[j]>0 else None,
                    'relative_l2_zero_denominator':bool(target_norm[j]==0),
                    'psnr':float(10*torch.log10(config['metrics']['data_range']**2/mse[j])) if mse[j]>0 else None,
                    'psnr_zero_mse':bool(mse[j]==0),'ssim':float(similarity[j]),
                    'data_change':float(change[j]),'data_residual':float(residual[j]),
                    'delta':float(b['delta'][j]),'beta':float(budget[j]) if constrained else None,
                    'control':float(control[j]) if constrained else None,'null_leak':float(leak[j]),
                    'signed_beta_excess':float(change[j]-budget[j]) if constrained else None,
                    'data_residual_over_delta':float(residual[j]/b['delta'][j]) if b['delta'][j]>0 else None,
                    'data_residual_over_delta_zero_denominator':bool(b['delta'][j]==0),
                    'leakage_bound':float(budget[j]+(1 if method=='DP_TWO' else 1-control[j])*leak[j]) if constrained else None}
                rows.append(row);archive.rows('metrics',[row])
                if constrained and row['data_change']>row['leakage_bound']+config['metrics']['proximity_atol']:
                    raise ArithmeticError('Per-object data proximity failed')
            predictions.append(array(out))
    path=archive.path/'predictions'/f'{label}_{step:08d}.npz';path.parent.mkdir(exist_ok=True)
    np.savez_compressed(path,ids=np.asarray(ids),images=np.concatenate(predictions))
    summary=aggregate(rows)
    write_json(archive.path/f'{label}_{step:08d}_summary.json',summary)
    archive.rows('evaluations',[{'evaluation_id':event_id,'status':'completed','summary':summary,
        'predictions':file_reference(path),'weight_reference':None if baseline else weight_reference}])
    archive.flush_metadata();return summary


SELECTION_CONTRACT = {
    'split': 'validation', 'aggregation': 'per_object_arithmetic_mean',
    'checkpoints': {'mse': 'min', 'ssim': 'max'},
    'ties': 'retain_first', 'early_stopping': 'mse_min_only',
    'min_delta_scope': 'early_stopping_only', 'legacy_best_alias': 'mse',
}


def update_best(progress, result, weights):
    """Select independently; patience is deliberately handled by the caller."""
    scores = {key: float(result[key]['mean']) for key in ('mse', 'ssim')}
    if not all(math.isfinite(v) for v in scores.values()):
        raise ArithmeticError('Nonfinite checkpoint-selection metric')
    for metric in ('mse', 'ssim'):
        key = 'best_' + metric
        previous = progress[key]
        improved = previous is None or (scores[metric] < previous[metric] if metric == 'mse'
                                        else scores[metric] > previous[metric])
        if improved:
            progress[key] = {**scores, 'epoch': progress['epoch'], 'step': progress['step'],
                             'weights': weights}
    progress['best'] = progress['best_mse']


def write_best(archive, progress):
    for name, key in (('best_model', 'best'), ('best_mse', 'best_mse'), ('best_ssim', 'best_ssim')):
        if progress[key] is not None:
            selected = copy.deepcopy(progress[key])
            source = verify_reference(selected['weights'])
            if source.parent.parent != archive.path:
                # Publish inherited best weights in this segment as well. A completed
                # resumed run must remain evaluable when its best predates the stop.
                import shutil
                destination = archive.path/'weights'/('inherited_'+selected['weights']['sha256']+'.pt')
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists():
                    with source.open('rb') as src, destination.open('xb') as dst:
                        shutil.copyfileobj(src, dst)
                        dst.flush(); os.fsync(dst.fileno())
                copied = file_reference(destination)
                if copied['sha256'] != selected['weights']['sha256']:
                    raise ValueError('Inherited best-weight publication changed content')
                selected['source_weights'] = selected['weights']
                selected['weights'] = copied
            write_json(archive.path/(name + '.json'), selected)


def finalized_config(config):
    """Materialize archive contracts identically for execution and transition audit."""
    config=copy.deepcopy(config)
    config['training']['applicable']=config['mode']=='train'
    config['checkpoint_selection']=copy.deepcopy(SELECTION_CONTRACT)
    if not config['selection'].get('early_stopping_enabled',True):
        config['checkpoint_selection']['early_stopping']='disabled_fixed_epoch_budget'
    config['dp_contract']={'variant':config['model']['method'],'control_norm':'A(I-P)V' if config['model']['method']=='DP_TWO' else 'AU','bound':'beta+norm(APU)' if config['model']['method']=='DP_TWO' else 'beta+(1-t)*norm(APU)'}
    config['retention_contract']={'mode':'every_step' if config['logging']['weights_every_step'] else 'periodic',
        'full_step_metadata':True,'all_evaluated_weights':True,'epoch_and_interrupt_checkpoints':True,
        'checkpoint_interval_steps':config['logging']['checkpoint_every_steps'],
        'unsaved_states':'identified by run/step; recover from last committed checkpoint, no arbitrary-step restoration guarantee'}
    config['model_evaluation']={'applicable':True,
        'reason':'Network evaluations use preserved weights and per-object records in evaluations.jsonl/metrics.jsonl'}
    return config


def scientific_configuration(config):
    scientific=copy.deepcopy({k:v for k,v in config.items() if k not in ('purpose','mode','execution','study_id')})
    scientific['training'].pop('applicable')
    return scientific


def execute(config,root):
    config=finalized_config(config)
    validate_config(config);root=Path(root)
    parent=None
    if config['execution']['resume']:
        parent=Path(config['execution']['resume']['path']).parent.parent.name
    os.environ['CUBLAS_WORKSPACE_CONFIG']=config['computation']['cublas_workspace_config']
    with TrainingArchive(config,root/'runs',SOURCE_ROOT,root/'metadata',parent_run_id=parent) as archive:
        print('Training segment: '+str(archive.path),flush=True)
        data=CachedData(config)
        write_json(archive.path/'dataset_binding.json',data.manifest)
        device=setup(archive,config);torch.set_num_threads(config['computation']['cpu_threads'])
        random.seed(config['randomness']['python_seed']);np.random.seed(config['randomness']['numpy_seed'])
        torch.manual_seed(config['randomness']['model_seed'])
        model=build_model(config,device,archive)
        kwargs={k:v for k,v in config['optimizer'].items() if k!='name'};kwargs['betas']=tuple(kwargs['betas'])
        optimizer=torch.optim.Adam(model.correction.parameters(),**kwargs)
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer,**{k:v for k,v in config['scheduler'].items() if k!='name'})
        sampler=torch.Generator(device='cpu').manual_seed(config['randomness']['sampler_seed'])
        manifest=json.loads((archive.path/'manifest.json').read_text())
        source_hash=canonical_hash([x for x in manifest['source_files'] if x['path'].endswith('.py')])
        scientific=scientific_configuration(config)
        binding={'configuration_sha256':canonical_hash(scientific),'source_python_sha256':source_hash,
                 'torch':str(torch.__version__),'numpy':np.__version__,'cuda':torch.version.cuda,
                 'device_name':torch.cuda.get_device_name(device) if device.type=='cuda' else 'cpu',
                 'projection':model.projection.contract}
        write_json(archive.path/'resume_binding.json',binding)
        progress={'step':0,'epoch':0,'cursor':0,'order':None,'phase':'train','best':None,'best_mse':None,'best_ssim':None,
                  'early_stopping_reference':None,'bad_epochs':0}
        if config['execution']['resume']:
            ref=config['execution']['resume'];path=verify_reference(ref)
            state=torch.load(path,map_location=device,weights_only=True)
            transition=None
            if state['binding']!=binding:
                from dpnn_experiments.epoch_extension import validate_transition
                transition=validate_transition(config,state,binding,ref)
            wp=verify_reference(state['weight_reference'])
            model.load_state_dict(torch.load(wp,map_location=device,weights_only=True))
            optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
            # PyTorch 2.7 StepLR's first-call warning marker is not part of Adam's
            # state_dict. Restore it too when resuming just before validation.
            # The checkpoint binding pins the exact PyTorch version.
            optimizer._opt_called=state['optimizer_runtime']['step_called']
            progress=state['progress'];restore_rng(state['rng'],sampler,device)
            if transition is not None:
                # Only a sealed budget-exhausted parent may be reopened. No state
                # is reset: global epoch/step, Adam, RNG, bests and patience survive.
                progress['phase']='train'
                archive.event('epoch_budget_extended',transition=config['execution']['continuation'],
                              from_epochs=transition['from_epochs'],to_epochs=transition['to_epochs'],
                              early_stopping_enabled=False)
            for key in ('best_mse','best_ssim'):
                if progress[key] is not None:verify_reference(progress[key]['weights'])
            archive.event('resume_restored',checkpoint=ref,position=progress)
        initial_weights=archive.weights(model,progress['step'])
        archive.checkpoint(model,optimizer,scheduler,sampler,progress,binding,'segment_start')
        if config['mode']=='evaluate':
            if not config['execution']['resume']:raise ValueError('Standalone evaluation requires a checkpoint')
            evaluate(archive,config,model,data,initial_weights,progress['step'],'standalone')
            write_json(archive.path/'summary.json',{'kind':'standalone_validation','step':progress['step']})
            return archive.path
        if progress['step']==0:
            evaluate(archive,config,model,data,None,0,'baseline',baseline=True)
        requested={'signal':None}
        old_handlers={}
        def on_signal(number,frame):requested['signal']=number
        for number in (signal.SIGTERM,signal.SIGINT):
            old_handlers[number]=signal.signal(number,on_signal)
        try:
            while progress['epoch']<config['training']['epochs'] and progress['phase']!='done':
                if progress['phase']=='train':
                    if progress['order'] is None:
                        ids=data.used['train'];permutation=torch.randperm(len(ids),generator=sampler).tolist()
                        progress['order']=[ids[i] for i in permutation]
                        archive.event('epoch_order',epoch=progress['epoch'],ids=progress['order'])
                    names=progress['order'][progress['cursor']:progress['cursor']+config['training']['batch_size']]
                    model.train();b=data.batch(names,device,split='train')
                    before=archive.step_weights(model,progress['step'],config['logging']['weights_every_step']);began=time.perf_counter()
                    optimizer.zero_grad(set_to_none=True)
                    out=forward(model,b,config['model']['method'])
                    per_object=(out-b['target']).square().flatten(1).mean(1);loss=per_object.mean()
                    if not bool(torch.isfinite(loss)):raise ArithmeticError('Nonfinite training loss')
                    loss.backward()
                    gradient_norm=torch.nn.utils.clip_grad_norm_(model.correction.parameters(),config['training']['gradient_clip_norm'],
                                                               norm_type=2.,error_if_nonfinite=True)
                    learning_rates=[g['lr'] for g in optimizer.param_groups];optimizer.step()
                    if device.type=='cuda':torch.cuda.synchronize(device)
                    numerical_seconds=time.perf_counter()-began
                    progress['step']+=1;progress['cursor']+=len(names)
                    if progress['cursor']==len(progress['order']):progress['phase']='validate'
                    after=archive.step_weights(model,progress['step'],config['logging']['weights_every_step'])
                    row={'step':progress['step'],'epoch':progress['epoch'],'ids':names,'loss':float(loss.detach()),
                        'per_object_loss':array(per_object).tolist(),'learning_rates':learning_rates,
                        'beta':([float(v) for v in b['beta']] if config['model']['method'] in ('DP','DP_TWO') else
                                [0. if config['model']['method']=='NSN' else None for v in names]),
                        'cached_beta':[float(v) for v in b['beta']],'delta':[float(v) for v in b['delta']],
                        'gradient_norm_before_clip':float(gradient_norm),'weight_before':before,'weight_after':after,
                        'recovery_checkpoint':archive.last_checkpoint,
                        'numerical_seconds':numerical_seconds,'host_max_rss_native_units':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                        'gpu_allocated_bytes':torch.cuda.memory_allocated(device) if device.type=='cuda' else None,
                        'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None}
                    archive.rows('training_steps',[row]);archive.event('optimizer_step',step=progress['step'],loss=row['loss'])
                    if config['execution']['injected_io_failure_step']==progress['step']:
                        raise OSError('Injected logging write failure for qualification; no further optimizer step')
                    stop=(config['execution']['stop_after_step'] is not None and progress['step']>=config['execution']['stop_after_step']) or requested['signal']
                    if progress['step']%config['logging']['checkpoint_every_steps']==0 or stop:
                        archive.checkpoint(model,optimizer,scheduler,sampler,progress,binding,'train')
                    if stop:
                        archive.outcome='interrupted';archive.reason='signal '+str(requested['signal']) if requested['signal'] else 'planned checkpoint/resume qualification stop'
                        break
                if progress['phase']=='validate':
                    weights=archive.weights(model,progress['step'])
                    archive.checkpoint(model,optimizer,scheduler,sampler,progress,binding,'pre_validation')
                    result=evaluate(archive,config,model,data,weights,progress['step'],'validation')
                    score=result['mse']['mean']
                    update_best(progress,result,weights)
                    write_best(archive,progress)
                    if progress['early_stopping_reference'] is None or score<progress['early_stopping_reference']-config['selection']['min_delta']:
                        progress['early_stopping_reference']=score;progress['bad_epochs']=0
                    else:progress['bad_epochs']+=1
                    progress['phase']='advance'
                if progress['phase']=='advance':
                    scheduler.step();progress['epoch']+=1;progress['cursor']=0;progress['order']=None
                    early_stop=(config['selection'].get('early_stopping_enabled',True)
                                and progress['bad_epochs']>=config['selection']['patience'])
                    progress['phase']='done' if (progress['epoch']>=config['training']['epochs'] or early_stop) else 'train'
                    archive.checkpoint(model,optimizer,scheduler,sampler,progress,binding,'epoch_end')
                    print(f"Epoch {progress['epoch']}, step {progress['step']}, best validation MSE {progress['best']['mse']:.8g}",flush=True)
        finally:
            for number,handler in old_handlers.items():signal.signal(number,handler)
        write_json(archive.path/'summary.json',{'position':progress,'last_checkpoint':archive.last_checkpoint,
            'best_model':progress['best'],'best_mse':progress['best_mse'],'best_ssim':progress['best_ssim'],'status':archive.outcome,'test_read_count':sum(x in data.manifest['split_ids']['test'] for x in data.read_ids),
            'read_ids':data.read_ids,'training_profile':'single device Float64, fixed cached noise, no augmentation or AMP'})
        write_best(archive,progress)
    return archive.path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--audit',action='store_true')
    a=p.parse_args()
    if a.audit:print(json.dumps(audit_runs(a.root),indent=2))
    else:execute(json.loads(a.config.read_text()),a.root)


if __name__=='__main__':main()
