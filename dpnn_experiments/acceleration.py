"""Fixed Chebyshev filters and left row scaling in whitened coordinates."""
import math
import numpy as np
import torch
from dpnn_experiments.projections import MatrixOperator


def chebyshev_native(value, operator, iterations, lower, upper):
    if iterations == 0:
        return value
    c=(upper+lower)/(upper-lower)
    d=2/(upper-lower)
    previous=value
    current=value-(d/c)*operator.transpose(operator.forward(value))
    ratio=1/c
    for _ in range(1,iterations):
        denominator=2*c-ratio
        following=(2*(c*current-d*operator.transpose(operator.forward(current)))-ratio*previous)/denominator
        previous,current=current,following
        ratio=1/denominator
    return current


class _Chebyshev(torch.autograd.Function):
    @staticmethod
    def forward(ctx,value,operator,iterations,lower,upper):
        ctx.constants=(operator,iterations,lower,upper)
        return chebyshev_native(value,*ctx.constants)

    @staticmethod
    def backward(ctx,gradient):
        return _Chebyshev.apply(gradient,*ctx.constants),None,None,None,None


class FixedChebyshev:
    """P_K=T_K((b+a-2 M.T M)/(b-a))/T_K((b+a)/(b-a)).

The interval [a,b] is an explicit spectral design choice, not an estimated
positive-spectrum enclosure. Zero modes have gain one; modes below a persist.
The backward is the same self-adjoint finite polynomial.
"""
    def __init__(self,operator,iterations,lower,upper=None):
        upper=operator.norm_bound_squared if upper is None else float(upper)
        lower=float(lower)
        if type(iterations) is not int or iterations<0 or not (math.isfinite(lower) and math.isfinite(upper)):
            raise ValueError('Invalid iteration count or interval')
        if not 0<lower<upper or upper<operator.norm_bound_squared:
            raise ValueError('Require 0<a<b and b >= the recorded squared norm bound')
        self.constants=(operator,iterations,lower,upper)

    def __call__(self,value):
        return _Chebyshev.apply(value,*self.constants)


def row_scaled_radon(radon,device):
    factor=math.sqrt(radon.geometry.data_weight)/(2/radon.geometry.size)
    values=radon.values*factor
    row_norm=np.sqrt(np.bincount(radon.rows,weights=values**2,minlength=radon.shape[0]))
    weights=np.ones_like(row_norm)
    np.divide(1.0,row_norm,out=weights,where=row_norm>0)
    scaled=values*weights[radon.rows]
    bound=float(np.bincount(radon.rows,weights=np.abs(scaled),minlength=radon.shape[0]).max()
                *np.bincount(radon.cols,weights=np.abs(scaled),minlength=radon.shape[1]).max())
    matrix=torch.sparse_coo_tensor(torch.from_numpy(np.stack((radon.rows,radon.cols)).astype(np.int64)),
                                   torch.from_numpy(scaled),radon.shape).coalesce().to(device)
    return MatrixOperator(matrix,bound),torch.tensor(weights,device=device)
