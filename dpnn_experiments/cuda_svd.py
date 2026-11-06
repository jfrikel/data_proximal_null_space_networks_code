# SPDX-License-Identifier: GPL-3.0-only
"""Minimal Float64 binding to the documented cuSOLVER 64-bit Xgesvd API.

The input is a tall column-major CUDA tensor and is overwritten. No autograd
through this factorization is provided. Result factors can be held fixed later.
"""
import ctypes as C
from ctypes.util import find_library
import os
import numpy as np
import torch


class XGesvd:
    def __init__(self, vectors=True):
        self.vectors = bool(vectors)
        library = os.environ.get('DPNN_CUSOLVER_LIBRARY') or find_library('cusolver')
        if not library:
            raise RuntimeError('cuSOLVER not found; set DPNN_CUSOLVER_LIBRARY to your installed library path')
        self.lib=C.CDLL(library)
        self.handle=C.c_void_p();self.params=C.c_void_p()
        for name,args in (
            ('cusolverDnCreate',[C.POINTER(C.c_void_p)]),
            ('cusolverDnCreateParams',[C.POINTER(C.c_void_p)]),
            ('cusolverDnSetStream',[C.c_void_p,C.c_void_p]),
            ('cusolverDnDestroyParams',[C.c_void_p]),('cusolverDnDestroy',[C.c_void_p])):
            f=getattr(self.lib,name);f.argtypes=args;f.restype=C.c_int
        prefix=[C.c_void_p,C.c_void_p,C.c_byte,C.c_byte,C.c_int64,C.c_int64,
                C.c_int,C.c_void_p,C.c_int64,C.c_int,C.c_void_p,
                C.c_int,C.c_void_p,C.c_int64,C.c_int,C.c_void_p,C.c_int64,C.c_int]
        self.lib.cusolverDnXgesvd_bufferSize.argtypes=prefix+[C.POINTER(C.c_size_t),C.POINTER(C.c_size_t)]
        self.lib.cusolverDnXgesvd.argtypes=prefix+[C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t,C.c_void_p]
        self.lib.cusolverDnXgesvd_bufferSize.restype=C.c_int
        self.lib.cusolverDnXgesvd.restype=C.c_int
        self.check(self.lib.cusolverDnCreate(C.byref(self.handle)),'create')
        self.check(self.lib.cusolverDnCreateParams(C.byref(self.params)),'create_params')
        self.check(self.lib.cusolverDnSetStream(self.handle,C.c_void_p(torch.cuda.current_stream().cuda_stream)),'set_stream')

    @staticmethod
    def check(status,name):
        if status:raise RuntimeError(f'cuSOLVER {name} status={status}')

    def close(self):
        self.lib.cusolverDnDestroyParams(self.params)
        self.lib.cusolverDnDestroy(self.handle)

    def prefix(self,m,n,a,s,u,vt):
        ptr=lambda x:C.c_void_p(x.data_ptr())
        # CUDA_R_64F == 1; job S returns reduced singular vectors.
        job=ord('S') if self.vectors else ord('N')
        return [self.handle,self.params,job,job,m,n,1,ptr(a),m,
                1,ptr(s),1,ptr(u),m,1,ptr(vt),n,1]

    def query(self,m,n):
        dummy=torch.empty(1,device='cuda',dtype=torch.float64)
        dev=C.c_size_t();host=C.c_size_t()
        status=self.lib.cusolverDnXgesvd_bufferSize(*self.prefix(m,n,dummy,dummy,dummy,dummy),
                                                  C.byref(dev),C.byref(host))
        return {'api':'cusolverDnXgesvd_bufferSize','status':status,
                'device_workspace_bytes':dev.value,'host_workspace_bytes':host.value,
                'workspace_query_uses_non_null_placeholder_pointers':True}

    def factor(self,a,callback=None):
        if a.dtype!=torch.float64 or a.ndim!=2 or not a.is_cuda or a.requires_grad:
            raise ValueError('Require fixed Float64 CUDA matrix')
        m,n=a.shape
        if m<n or a.stride()!=(1,m):raise ValueError('Require tall column-major contiguous input')
        s=torch.empty(n,device=a.device,dtype=a.dtype)
        u=torch.empty((n,m) if self.vectors else (1,1),device=a.device,dtype=a.dtype).T
        vt=torch.empty((n,n) if self.vectors else (1,1),device=a.device,dtype=a.dtype).T
        prefix=self.prefix(m,n,a,s,u,vt)
        dev=C.c_size_t();host=C.c_size_t()
        self.check(self.lib.cusolverDnXgesvd_bufferSize(*prefix,C.byref(dev),C.byref(host)),'buffer_size')
        if callback:callback({'device_workspace_bytes':dev.value,'host_workspace_bytes':host.value})
        device_buffer=torch.empty(dev.value,device=a.device,dtype=torch.uint8)
        host_buffer=np.empty(max(1,host.value),dtype=np.uint8)
        info=torch.zeros(1,device=a.device,dtype=torch.int32)
        self.check(self.lib.cusolverDnXgesvd(*prefix,C.c_void_p(device_buffer.data_ptr()),dev.value,
                      C.c_void_p(host_buffer.ctypes.data),host.value,C.c_void_p(info.data_ptr())),'factor')
        torch.cuda.synchronize(a.device)
        value=int(info.item())
        if value:raise ArithmeticError(f'cuSOLVER Xgesvd info={value}')
        return u,s,vt


class Xsyevd(XGesvd):
    """64-bit symmetric eigensolver, overwriting a column-major Float64 matrix."""
    def __init__(self):
        super().__init__()
        prefix=[C.c_void_p,C.c_void_p,C.c_int,C.c_int,C.c_int64,C.c_int,C.c_void_p,
                C.c_int64,C.c_int,C.c_void_p,C.c_int]
        self.lib.cusolverDnXsyevd_bufferSize.argtypes=prefix+[C.POINTER(C.c_size_t),C.POINTER(C.c_size_t)]
        self.lib.cusolverDnXsyevd.argtypes=prefix+[C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t,C.c_void_p]
        self.lib.cusolverDnXsyevd_bufferSize.restype=C.c_int
        self.lib.cusolverDnXsyevd.restype=C.c_int

    def eigen_prefix(self,n,a,w):
        # CUSOLVER_EIG_MODE_VECTOR=1; CUBLAS_FILL_MODE_LOWER=0.
        return [self.handle,self.params,1,0,n,1,C.c_void_p(a.data_ptr()),n,1,C.c_void_p(w.data_ptr()),1]

    def query(self,n):
        dummy=torch.empty(1,device='cuda',dtype=torch.float64)
        dev=C.c_size_t();host=C.c_size_t()
        status=self.lib.cusolverDnXsyevd_bufferSize(*self.eigen_prefix(n,dummy,dummy),C.byref(dev),C.byref(host))
        return {'api':'cusolverDnXsyevd_bufferSize','status':status,'device_workspace_bytes':dev.value,
                'host_workspace_bytes':host.value}

    def factor(self,a,callback=None):
        if a.dtype!=torch.float64 or a.ndim!=2 or not a.is_cuda or a.requires_grad:
            raise ValueError('Require fixed Float64 CUDA matrix')
        n=a.shape[0]
        if a.shape!=(n,n) or a.stride()!=(1,n):raise ValueError('Require square column-major matrix')
        w=torch.empty(n,device=a.device,dtype=a.dtype)
        prefix=self.eigen_prefix(n,a,w)
        dev=C.c_size_t();host=C.c_size_t()
        self.check(self.lib.cusolverDnXsyevd_bufferSize(*prefix,C.byref(dev),C.byref(host)),'eigen_buffer_size')
        if callback:callback({'device_workspace_bytes':dev.value,'host_workspace_bytes':host.value})
        device_buffer=torch.empty(dev.value,device=a.device,dtype=torch.uint8)
        host_buffer=np.empty(max(1,host.value),dtype=np.uint8)
        info=torch.zeros(1,device=a.device,dtype=torch.int32)
        self.check(self.lib.cusolverDnXsyevd(*prefix,C.c_void_p(device_buffer.data_ptr()),dev.value,
                      C.c_void_p(host_buffer.ctypes.data),host.value,C.c_void_p(info.data_ptr())),'eigen_factor')
        torch.cuda.synchronize(a.device)
        if int(info.item()):raise ArithmeticError(f'cuSOLVER Xsyevd info={int(info.item())}')
        return w,a
