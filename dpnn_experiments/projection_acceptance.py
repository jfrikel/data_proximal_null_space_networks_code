# SPDX-License-Identifier: GPL-3.0-only
"""Reusable Float64 projection acceptance; exact reference only on small fixtures."""
import torch
from dpnn_experiments.provenance import write_json


def accept_projection(projection,archive,seed=3601):
    p=projection;n=p.basis.shape[0];generator=torch.Generator(device=p.basis.device).manual_seed(seed)
    x=torch.randn((3,n),generator=generator,device=p.basis.device,dtype=torch.float64)
    x=x/torch.linalg.vector_norm(x,dim=1,keepdim=True)*x.new_tensor([1e-4,1.,1e4])[:,None]
    y=torch.randn(x.shape,generator=generator,device=x.device,dtype=x.dtype)
    px=p(x);py=p(y)
    norms=torch.linalg.vector_norm(x,dim=1)
    idem=torch.linalg.vector_norm(p(px)-px,dim=1)/norms
    symmetry=abs((px*y).sum()-(x*py).sum())/(torch.linalg.vector_norm(x)*torch.linalg.vector_norm(y))
    leak=torch.linalg.vector_norm(p.data(px),dim=1)/norms
    u=x.detach().clone().requires_grad_(True)
    back=torch.autograd.grad((p(u)*y).sum(),u)[0]
    backward=torch.linalg.vector_norm(back-py)/torch.linalg.vector_norm(y)
    # Independently check A against its transpose including physical-coordinate scale.
    d=torch.randn(p.data(x).shape,generator=generator,device=x.device,dtype=x.dtype)
    mt=p.matrix.transpose(0,1)
    adj=(torch.sparse.mm(mt,d.T) if p.matrix.is_sparse else mt@d.T).T*p.data_scale
    adjoint=abs((p.data(x)*d).sum()-(x*adj).sum())/(torch.linalg.vector_norm(x)*torch.linalg.vector_norm(d))
    result={'seed':seed,'contract':p.contract,'idempotence_relative':idem.tolist(),
        'self_adjoint_relative':float(symmetry),'data_leak_per_input_norm':leak.tolist(),
        'backward_relative':float(backward),'adjoint_absolute_normalized':float(adjoint),
        'tolerances':{'projector':1e-9,'backward':1e-12,'adjoint':1e-12,'data_leak':1e-10},
        'global_evidence':'Pinned global block verification in contract; probe tests supplement it, do not prove exact algebraic rank'}
    result['passed']=bool((idem<1e-9).all() and symmetry<1e-9 and (leak<1e-10).all() and backward<1e-12 and adjoint<1e-12)
    write_json(archive.path/'projection_acceptance.json',result)
    if not result['passed']:raise ArithmeticError('Projection acceptance failed')
    return result
