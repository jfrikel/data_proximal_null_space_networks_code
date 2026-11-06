# SPDX-License-Identifier: GPL-3.0-only
"""Regularized inverse and isotropic TV in whitened image/data coordinates.

q=h*x, h=2/N. TV(x)=h*sum sqrt((Dx)^2+(Dy)^2)=sum |Dq|.
Forward differences have zero terminal differences (Neumann boundary).
No positivity, clipping, circular mask or extra L2 penalty is imposed.
"""
import math
import torch
from dpnn_experiments.pilot_data import tikhonov


def gradient(x):
    out=x.new_zeros((*x.shape,2))
    out[:-1,:,0]=x[1:,:]-x[:-1,:]
    out[:,:-1,1]=x[:,1:]-x[:,:-1]
    return out


def gradient_adjoint(p):
    out=p.new_zeros(p.shape[:2])
    out[:-1,:]-=p[:-1,:,0];out[1:,:]+=p[:-1,:,0]
    out[:,:-1]-=p[:,:-1,1];out[:,1:]+=p[:,:-1,1]
    return out


def ball(p,radius):
    return p/torch.clamp(torch.linalg.vector_norm(p,dim=-1,keepdim=True)/radius,min=1.)


def reconstruct(operator,y,size,settings):
    if y.ndim!=1 or not torch.isfinite(y).all():raise ValueError('Finite single measurement required')
    if settings['kind']=='tikhonov':
        if set(settings)!={'kind','alpha','rtol','atol','max_iterations'}:raise ValueError('Invalid Tikhonov settings')
        x,info=tikhonov(operator,y[:,None],**{k:v for k,v in settings.items() if k!='kind'})
        return x[:,0].reshape(size,size)/(2/size),info
    if settings['kind']!='tv':raise ValueError('Unknown start method')
    if set(settings)!={'kind','alpha','rtol','atol','max_iterations','check_every','step_safety'}:raise ValueError('Invalid TV settings')
    alpha=settings['alpha'];rtol=settings['rtol'];atol=settings['atol'];limit=settings['max_iterations'];every=settings['check_every']
    if not all(math.isfinite(v) and v>0 for v in (alpha,rtol,atol)) or not 0<settings['step_safety']<1 or limit<1 or every<1:raise ValueError('Invalid TV parameters')
    step=settings['step_safety']/math.sqrt(operator.norm_bound_squared+8.)
    x=y.new_zeros((size,size));ex=x.clone();p=torch.zeros_like(y);q=gradient(x);history=[];passed=False
    normy=float(torch.linalg.vector_norm(y));normaty=float(torch.linalg.vector_norm(operator.transpose(y)))
    for i in range(1,limit+1):
        p=(p+step*(operator.forward(ex.flatten())-y))/(1+step)
        q=ball(q+step*gradient(ex),alpha)
        new=x-step*(operator.transpose(p).reshape(size,size)+gradient_adjoint(q))
        ex=2*new-x;x=new
        if i%every==0 or i==limit:
            residual=operator.forward(x.flatten())-y;dx=gradient(x)
            stationary=torch.linalg.vector_norm(operator.transpose(p).reshape(size,size)+gradient_adjoint(q))
            data_kkt=torch.linalg.vector_norm(p-residual)
            tv_kkt=torch.linalg.vector_norm(q-ball(q+dx,alpha))
            thresholds=[atol+rtol*normaty,atol+rtol*normy,atol+rtol*alpha*size]
            errors=[float(stationary),float(data_kkt),float(tv_kkt)]
            objective=float(.5*residual.square().sum()+alpha*torch.linalg.vector_norm(dx,dim=-1).sum())
            passed=all(a<=b for a,b in zip(errors,thresholds))
            history.append({'iteration':i,'objective':objective,'kkt_residuals':errors,'thresholds':thresholds,'passed':passed})
            if not math.isfinite(objective):raise ArithmeticError('Nonfinite TV iterate')
            if passed:break
    return x/(2/size),{'passed':passed,'iterations':i,'history':history,'step':step,
        'operator_norm_squared_upper_bound':operator.norm_bound_squared,'gradient_norm_squared_bound':8.,
        'criterion':'KKT stationarity, data-dual consistency, TV-dual projection residual; not an image-error certificate',
        'objective':'0.5||M q-y_tilde||² + alpha sum isotropic|Dq|; q=h*x; unconstrained Neumann TV'}
