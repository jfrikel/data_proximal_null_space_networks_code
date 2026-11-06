"""Fixed-basis two-U-Net acceptance: proximity, independent VJP and directions."""
import argparse,copy,json,os,time
from pathlib import Path
import numpy as np
import torch
from torch.func import functional_call
from dpnn_experiments.training import build_model, build_correction_model
from dpnn_experiments.network_layers import FixedBlockProjection
from dpnn_experiments.training_archive import TrainingArchive
from dpnn_experiments.projection_diagnostics import setup, array
from dpnn_experiments.projection_acceptance import accept_projection
from dpnn_experiments.provenance import write_json
C=Path(__file__).resolve().parents[1]

def require(ok,message):
 if not ok:raise ArithmeticError(message)

def independent_vjp(a,name,model,z,beta,probe,tol):
 p=model.projection;z=z.detach().requires_grad_();params=dict(model.correction.named_parameters())
 out=model(z,beta);observed=torch.autograd.grad((out*probe).sum(),(z,*params.values()))
 u=model.correction['null'](z);v=model.correction['range'](z)
 with torch.no_grad():
  w=probe.flatten(1).T;vc=v.flatten(1).T;Q=p.basis;R=lambda x:Q@(Q.T@x)
  r=R(vc);ar=p.data_scale*(torch.sparse.mm(p.matrix,r) if p.matrix.is_sparse else p.matrix@r)
  norm=ar.norm(dim=0);b=torch.as_tensor(beta,device=z.device,dtype=z.dtype).expand_as(norm)
  active=(b>0)&(norm>b);safe=torch.where(active,norm,torch.ones_like(norm))
  t=torch.where(b==0,torch.zeros_like(b),torch.where(active,b/safe,torch.ones_like(b)))
  at=p.matrix.T;atar=p.data_scale*(torch.sparse.mm(at,ar) if at.is_sparse else at@ar)
  gu=w-R(w)
  gv=t[None,:]*R(w)-torch.where(active,b/safe**3,torch.zeros_like(b))[None,:]*(r*w).sum(0,keepdim=True)*R(atar)
  gu=gu.T.reshape_as(u);gv=gv.T.reshape_as(v)
 propagated=torch.autograd.grad((u*gu).sum()+(v*gv).sum(),(z,*params.values()))
 expected=(propagated[0]+probe,*propagated[1:]);rows=[]
 for key,actual,ref in zip(['input',*params],observed,expected):
  err=float((actual-ref).norm());scale=float(ref.norm());passed=err<=tol['vjp_atol']+tol['vjp_rtol']*scale
  rows.append({'tensor':key,'error':err,'reference_norm':scale,'passed':passed})
 write_json(a.path/(name+'_vjp.json'),rows)
 np.savez_compressed(a.path/(name+'_vjp.npz'),input_gradient=array(observed[0]),expected_input_gradient=array(expected[0]),upstream=array(probe),null_upstream=array(gu),range_upstream=array(gv))
 require(all(r['passed'] for r in rows),'Independent two-branch layer VJP mismatch')
 branch_norms={branch:sum(float(g.square().sum()) for key,g in zip(params,observed[1:]) if key.startswith(branch+'.'))**.5 for branch in ('null','range')}
 require(all(np.isfinite(x) for x in branch_norms.values()),'Nonfinite branch gradient')
 if float(beta)==0:require(branch_norms['range']==0,'Zero budget must eliminate V gradient')
 else:require(all(x>0 for x in branch_norms.values()),'Missing branch gradient')
 return {'name':name,'max_tensor_error':max(r['error'] for r in rows),'branch_norms':branch_norms,'passed':True}

def directions(a,model,z,beta,probe,config):
 z=z.detach().requires_grad_();params=dict(model.named_parameters())
 selected=['correction.null.outc.conv.weight','correction.range.outc.conv.weight']
 out=model(z,beta);grads=torch.autograd.grad((out*probe).sum(),(z,*(params[k] for k in selected)))
 rng=np.random.default_rng(config['randomness']['direction_seed']);records=[];payload={}
 for name,grad in zip(['input',*selected],grads):
  target=z if name=='input' else params[name];direction=torch.tensor(rng.normal(size=tuple(target.shape)),device=z.device,dtype=z.dtype);direction/=direction.norm()
  payload[name]=array(direction);analytic=float((grad*direction).sum());values=[]
  for eps in config['finite_difference_steps']:
   with torch.no_grad():
    if name=='input':plus=model(z+eps*direction,beta);minus=model(z-eps*direction,beta)
    else:plus=functional_call(model,{name:target+eps*direction},(z,beta));minus=functional_call(model,{name:target-eps*direction},(z,beta))
    numeric=float(((plus-minus)*probe).sum()/(2*eps))
   error=abs(analytic-numeric);tol=config['tolerances']['fd_atol']+config['tolerances']['fd_rtol']*max(abs(analytic),abs(numeric))
   values.append({'eps':eps,'numeric':numeric,'error':error,'tolerance':tol,'passed':error<=tol})
  records.append({'tensor':name,'analytic':analytic,'values':values,'passed':all(r['passed'] for r in values[-2:])})
 np.savez_compressed(a.path/'directions.npz',**payload)
 write_json(a.path/'finite_differences.json',records)
 require(all(r['passed'] for r in records),'Two finest finite differences did not pass')
 return records

def run(config,root):
 root=Path(root);os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
 with TrainingArchive(config,root/'runs',C,root/'metadata') as a:
  print('Acceptance archive:',a.path,flush=True);device=setup(a,config);torch.set_num_threads(4);size=config['geometry']['size']
  if config['projection']['kind']=='generated_fixture':
   rng=np.random.default_rng(config['randomness']['seed']);q,_=np.linalg.qr(rng.normal(size=(size*size,16)));mat=rng.normal(size=(24,16))@q.T
   np.savez_compressed(a.path/'operator_fixture.npz',basis=q,matrix=mat)
   p=FixedBlockProjection(torch.tensor(q,device=device),torch.tensor(mat,device=device),coordinates='physical',image_weight=(2/size)**2,contract={'kind':'generated_fixture','geometry':config['geometry']})
   model=build_correction_model(config,p,device)
  else:model=build_model(config,device,a);p=model.projection
  model.eval();weights=a.weights(model,0);accept_projection(p,a,config['randomness']['seed'])
  generator=np.random.default_rng(config['randomness']['input_seed'])
  gaussian=torch.tensor(generator.normal(size=(1,1,size,size)),device=device,dtype=torch.float64)
  coords=(np.arange(size)+.5)*2/size-1;circle=torch.tensor((coords[:,None]**2+coords[None,:]**2<.6**2).astype(float)[None,None],device=device)
  probe=torch.tensor(generator.normal(size=tuple(gaussian.shape)),device=device,dtype=torch.float64);probe/=probe.norm()
  np.savez_compressed(a.path/'inputs.npz',gaussian=array(gaussian),circle=array(circle),upstream=array(probe))
  records=[];vjps=[];betas={};began=time.perf_counter()
  for name,z in [('gaussian',gaussian),('circle',circle)]:
   with torch.no_grad():norm=p.data(p.components(model.correction['range'](z))[1]).norm(dim=1)
   for fraction in config['budget_fractions']:
    beta=norm*fraction;tag=f'{name}_beta{fraction:g}';betas[tag]=array(beta).tolist()
    with torch.no_grad():
     out,d=model(z,beta,return_details=True);change=p.data(out-z).norm(dim=1);ell=p.data(d['null']).norm(dim=1);bound=beta+ell
     tol=config['tolerances']['bound_atol']+config['tolerances']['bound_rtol']*torch.maximum(torch.maximum(p.data(out).norm(dim=1),p.data(z).norm(dim=1)),bound)
     row={'name':tag,'beta':array(beta).tolist(),'control':array(d['control']).tolist(),'data_change':array(change).tolist(),'null_leak':array(ell).tolist(),'bound':array(bound).tolist(),'signed_beta_excess':array(change-beta).tolist(),'signed_bound_excess':array(change-bound).tolist(),'tolerance':array(tol).tolist(),'passed':bool((change<=bound+tol).all())}
     np.savez_compressed(a.path/(tag+'.npz'),output=array(out),null_value=array(d['null_value']),value=array(d['value']),null=array(d['null']),range=array(d['range']),beta=array(beta))
     a.rows('proximity',[row]);require(row['passed'],'Data proximity bound failed');records.append(row)
    # Independent full parameter VJP also at the circle's network kinks.
    vjps.append(independent_vjp(a,tag,model,z,beta,probe,config['tolerances']))
    print(tag,'proximity/VJP passed',flush=True)
    if name=='gaussian' and fraction==.3:fd_beta=beta.clone()
  finite=directions(a,model,gaussian,fd_beta,probe,config)
  require(all(not b.requires_grad and b.grad is None for b in p.buffers()),'Projection unexpectedly differentiated')
  write_json(a.path/'summary.json',{'passed':True,'weights':weights,'parameters':sum(x.numel() for x in model.parameters()),'proximity':records,'independent_vjps':vjps,'finite_differences':finite,'betas':betas,'seconds':time.perf_counter()-began,'gpu_peak_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0,'test_read_count':0,'training':False,'gradient_scope':'Full independent analytic layer VJP for input and every parameter; finite differences for Gaussian input and output-convolution directions; no differentiability assertion at circle kinks'})
  print('Acceptance passed:',a.path,flush=True)
 return a.path

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--root',type=Path,required=True);args=ap.parse_args();run(json.loads(args.config.read_text()),args.root)
