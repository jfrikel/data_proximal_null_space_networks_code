"""Archived independent two-branch projection/gradient and actual U-Net checks."""
import argparse,copy,json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.func import functional_call
from dpnn_experiments.network_layers import FixedBlockProjection, SeparateDataProximalNetwork, ScalarDataProximalNetwork
from dpnn_experiments.training_archive import TrainingArchive
from dpnn_experiments.provenance import write_json
C=Path(__file__).resolve().parents[1]

def run(root):
 root=Path(root);config={'schema_version':1,'purpose':'J038 separate U/V projection and gradients; synthetic technical fixtures only',
 'dataset':{'kind':'seeded dense small SVD plus 32-square U-Net fixture; no scientific data'},'geometry':{'sizes':[3,32]},
 'randomness':{'seed':3801},'computation':{'device':'cpu','dtype':'float64','threads':4},
 'tolerances':{'bound_atol':1e-12,'gradcheck_atol':2e-7,'gradcheck_rtol':2e-5,'direction_atol':2e-7},'training':{'applicable':False}}
 with TrainingArchive(config,root/'runs',C,root/'metadata') as a:
  torch.set_num_threads(4);torch.manual_seed(3801);rng=np.random.default_rng(3801);results={}
  q,_=np.linalg.qr(rng.normal(size=(9,4)));left=rng.normal(size=(6,4));mat=left@q.T
  p=FixedBlockProjection(torch.tensor(q),torch.tensor(mat),coordinates='physical',image_weight=.04,contract={'fixture':'small_SVD'})
  u=nn.Linear(9,9).double();v=nn.Linear(9,9).double();m=SeparateDataProximalNetwork(u,v,p)
  a.weights(m,0);z=torch.randn(3,9,dtype=torch.float64,requires_grad=True)
  with torch.no_grad():r=p.data(p.components(v(z))[1]).norm(dim=1);beta=r*torch.tensor([0.,.3,2.])
  out,d=m(z,beta,return_details=True);ell=p.data(d['null']).norm(dim=1);change=p.data(out-z).norm(dim=1)
  assert bool((change<=beta+ell+1e-12).all())
  pref=np.eye(9)-np.linalg.pinv(mat)@mat
  un=u(z).detach().numpy();vn=v(z).detach().numpy();rn=vn@(np.eye(9)-pref).T
  norms=np.linalg.norm(.2*rn@mat.T,axis=1);tn=np.minimum(1,beta.numpy()/norms)
  expected=z.detach().numpy()+un@pref.T+tn[:,None]*rn
  results['SVD_output_max_error']=float(np.max(abs(out.detach().numpy()-expected)));assert results['SVD_output_max_error']<1e-12
  params=dict(m.named_parameters());names=list(params)
  results['full_input_U_V_gradcheck']=torch.autograd.gradcheck(lambda x,*ps:functional_call(m,dict(zip(names,ps)),(x,beta)),(z,*params.values()),eps=1e-6,atol=2e-7,rtol=2e-5)
  results['mixed_budget_change']=change.detach().tolist();results['mixed_budget_bound']=(beta+ell).detach().tolist()
  # beta=0 removes V even at zero V; a large budget yields z+PU+(I-P)V.
  assert torch.equal(m(z,0.),z+p(u(z)))
  assert torch.allclose(m(z,1e9),z+p(u(z))+p.components(v(z))[1],atol=1e-12,rtol=1e-12)
  shared=ScalarDataProximalNetwork(u,p);equal=SeparateDataProximalNetwork(copy.deepcopy(u),copy.deepcopy(u),p)
  results['equal_branches_shared_error']=float((shared(z,beta)-equal(z,beta)).abs().max().detach());assert results['equal_branches_shared_error']<1e-12
  zero=nn.Linear(9,9).double();nn.init.zeros_(zero.weight);nn.init.zeros_(zero.bias)
  zm=SeparateDataProximalNetwork(copy.deepcopy(u),zero,p);oz=zm(z,0.);gz=torch.autograd.grad(oz.sum(),tuple(zm.correction['range'].parameters()),allow_unused=True)
  assert all(g is not None and torch.equal(g,torch.zeros_like(g)) for g in gz)
  for b in (-1.,float('nan'),torch.ones(2),torch.tensor(1.,requires_grad=True)):
   try:m(z,b)
   except ValueError:pass
   else:raise AssertionError('Invalid budget accepted')
  try:SeparateDataProximalNetwork(u,u,p)
  except ValueError:pass
  else:raise AssertionError('Shared parameters accepted')
  # Intentionally defective projector: the old shared bound must fail at t=1.
  bad=FixedBlockProjection(torch.zeros(9,0,dtype=torch.float64),torch.tensor(mat),coordinates='physical',image_weight=.04,contract={'fixture':'deliberately_wrong_nullspace'})
  bm=SeparateDataProximalNetwork(copy.deepcopy(u),copy.deepcopy(v),bad)
  bo,bd=bm(z,1e-8,return_details=True);leak=bad.data(bd['null']).norm(dim=1);bc=bad.data(bo-z).norm(dim=1)
  assert bool((bc<=1e-8+leak+1e-12).all()) and bool((bc>1e-8+(1-bd['control'])*leak+1e-12).any())
  results['defective_projection_new_bound_passes_old_bound_fails']=True
  np.savez_compressed(a.path/'small_arrays.npz',matrix=mat,basis=q,z=z.detach().numpy(),beta=beta.numpy(),output=out.detach().numpy(),reference=expected,defective_change=bc.detach().numpy(),defective_leak=leak.detach().numpy())
  # Real full-width U-Nets, distinct parameters, shared fixed projection. No training.
  from src.unet import UNet
  q,_=np.linalg.qr(rng.normal(size=(1024,16)));mat=rng.normal(size=(24,16))@q.T
  pp=FixedBlockProjection(torch.tensor(q),torch.tensor(mat),coordinates='physical',image_weight=(2/32)**2,contract={'fixture':'32-square-UNet','geometry':{'size':32}})
  um=SeparateDataProximalNetwork(UNet(1,1,True).double(),UNet(1,1,True).double(),pp).eval()
  weight=a.weights(um,1);x=torch.randn(1,1,32,32,dtype=torch.float64,requires_grad=True)
  with torch.no_grad():ub=.3*pp.data(pp.components(um.correction['range'](x))[1]).norm(dim=1)
  pred,dd=um(x,ub,return_details=True);probe=torch.randn_like(pred);probe/=probe.norm();loss=(pred*probe).sum();loss.backward()
  branch_norms={k:sum(float(t.grad.square().sum()) for t in branch.parameters())**.5 for k,branch in um.correction.items()}
  assert all(np.isfinite(v) and v>0 for v in branch_norms.values())
  directions={};eps=1e-5
  dx=torch.randn_like(x);dx/=dx.norm()
  with torch.no_grad():numeric=float(((um(x+eps*dx,ub)-um(x-eps*dx,ub))*probe).sum()/(2*eps))
  analytic=float((x.grad*dx).sum());directions['input']={'analytic':analytic,'numeric':numeric,'error':abs(analytic-numeric)}
  for branch in ('null','range'):
   name='correction.'+branch+'.outc.conv.weight';param=dict(um.named_parameters())[name];direction=torch.randn_like(param);direction/=direction.norm()
   with torch.no_grad():numeric=float(((functional_call(um,{name:param+eps*direction},(x,ub))-functional_call(um,{name:param-eps*direction},(x,ub)))*probe).sum()/(2*eps))
   analytic=float((param.grad*direction).sum());directions[branch]={'analytic':analytic,'numeric':numeric,'error':abs(analytic-numeric)}
  write_json(a.path/'unet_directional.json',directions)
  assert all(v['error']<2e-7 for v in directions.values())
  leak=pp.data(dd['null']).norm(dim=1);change=pp.data(pred-x).norm(dim=1)
  assert bool((change<=ub+leak+1e-12).all())
  results['UNet']={'size':32,'parameters':sum(p.numel() for p in um.parameters()),'branch_gradient_norms':branch_norms,'directions':directions,'weights':weight,'data_change':change.detach().tolist(),'budget':ub.tolist(),'leak':leak.detach().tolist()}
  np.savez_compressed(a.path/'unet_arrays.npz',matrix=mat,basis=q,x=x.detach().numpy(),beta=ub.numpy(),probe=probe.numpy(),output=pred.detach().numpy())
  results['passed']=True;write_json(a.path/'summary.json',results);print(json.dumps(results,indent=2))
 return a.path

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args();run(args.root)
