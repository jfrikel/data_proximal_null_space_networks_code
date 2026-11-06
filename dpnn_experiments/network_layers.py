# SPDX-License-Identifier: GPL-3.0-only
"""Float64 NSN/scalar-DP layers using fixed, provenance-bound row-space factors.

Images are batch-first: (B,n), (B,H,W), or (B,1,H,W), with H*W=n.
For physical pixel values beta is in the weighted measurement norm. For
whitened coordinates beta is in the Euclidean data norm. No output clipping.
"""
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from dpnn_experiments.gpu_benchmark import load_operator, reference_metadata
from dpnn_experiments.projections import MatrixOperator
from dpnn_experiments.provenance import digest


def checked_archive_file(root, name, expected=None):
    root = Path(root)
    index = {r['path']: r for r in json.loads((root/'artifacts.json').read_text())['files']}
    record = index[name]
    if expected is not None and record['sha256'] != expected:
        raise ValueError(f'Unexpected pinned hash for {name}')
    path = root/name
    if path.stat().st_size != record['bytes'] or digest(path) != record['sha256']:
        raise ValueError(f'Archive integrity failure: {name}')
    return path


class FixedBlockProjection(nn.Module):
    """P=I-Q Q.T and weighted forward operator, with fixed Float64 buffers.

Large buffers are excluded from state_dict. The exact contract is included;
loading a checkpoint requires reconstructing the identical verified factors.
Direct construction is for explicitly recorded small reference matrices.
"""
    def __init__(self, basis, matrix, *, coordinates, image_weight, contract):
        super().__init__()
        if coordinates not in ('physical', 'whitened'):
            raise ValueError('Specify physical or whitened coordinates')
        if not math.isfinite(image_weight) or image_weight <= 0:
            raise ValueError('Image weight must be positive and finite')
        if (basis.ndim != 2 or matrix.ndim != 2 or basis.shape[0] != matrix.shape[1]
                or basis.shape[1] > basis.shape[0] or matrix.shape[0] == 0):
            raise ValueError('Incompatible factor/operator shapes')
        for value in (basis, matrix):
            if value.dtype != torch.float64 or value.requires_grad or value.device != basis.device:
                raise ValueError('Fixed factors and operator require identical Float64 device')
            coefficients = value.coalesce().values() if value.layout == torch.sparse_coo else value
            if not bool(torch.isfinite(coefficients).all()):
                raise ValueError('Nonfinite factor/operator')
        if basis.layout != torch.strided or matrix.layout not in (torch.strided, torch.sparse_coo):
            raise ValueError('Require dense basis and dense/COO matrix')
        self.register_buffer('basis', basis, persistent=False)
        self.register_buffer('matrix', matrix.coalesce() if matrix.is_sparse else matrix, persistent=False)
        self.coordinates = coordinates
        self.image_weight = float(image_weight)
        self.data_scale = math.sqrt(image_weight) if coordinates == 'physical' else 1.0
        self.contract = copy.deepcopy(contract)
        self.contract.update(schema=1, coordinates=coordinates, image_weight=self.image_weight,
                             matrix_shape=list(matrix.shape), basis_shape=list(basis.shape), dtype='float64')

    @classmethod
    def from_archive(cls, factor_run, *, basis_sha256, operator_sha256, geometry,
                     verification_run, verification_summary_sha256, coordinates, device):
        factor_run, verification_run = Path(factor_run), Path(verification_run)
        for directory in (factor_run, verification_run):
            checked_archive_file(directory, 'manifest.json')
            checked_archive_file(directory, 'completion.json')
            if json.loads((directory/'completion.json').read_text())['status'] != 'completed':
                raise ValueError('Require successfully completed factor/verification runs')
        manifest = json.loads((factor_run/'manifest.json').read_text())
        if manifest['configuration']['mode'] != 'block_factorization':
            raise ValueError('Require a block factorization archive')
        reference = reference_metadata(factor_run/'reference')
        if reference['files']['operator.npz'] != operator_sha256 or reference['geometry'] != geometry:
            raise ValueError('Factor operator or geometry does not match requested contract')
        recorded = manifest['configuration']['dataset']['reference_operator']
        if recorded['files']['operator.npz'] != operator_sha256:
            raise ValueError('Basis was computed for a different operator')
        check_manifest = json.loads((verification_run/'manifest.json').read_text())
        if (check_manifest['configuration']['mode'] != 'factor_verification'
                or check_manifest['configuration']['dataset']['factor_basis']['sha256'] != basis_sha256
                or check_manifest['configuration']['dataset']['reference_operator']['files']['operator.npz'] != operator_sha256):
            raise ValueError('Global verification does not belong to these factors/operator')
        validation = json.loads(checked_archive_file(verification_run, 'summary.json',
                                                     verification_summary_sha256).read_text())
        if validation['MP_relative_frobenius_norm'] > 1e-10 or validation['QtQ_minus_I_frobenius_norm'] > 1e-8:
            raise ValueError('Archived projection does not meet diagnostic acceptance tolerances')
        file = checked_archive_file(factor_run, 'row_space_basis.npy', basis_sha256)
        loaded = np.load(file, allow_pickle=False)
        if loaded.dtype != np.float64:
            raise ValueError('Basis file must actually contain Float64')
        radon = load_operator(factor_run/'reference', reference)
        op = MatrixOperator.whitened_radon(radon, device)
        contract = {'kind': 'two_block_row_space', 'factor_run_id': manifest['run_id'],
                    'basis_sha256': basis_sha256, 'operator_sha256': operator_sha256,
                    'geometry': geometry, 'verification_run_id': check_manifest['run_id'],
                    'verification_summary_sha256': verification_summary_sha256,
                    'block_cutoff': manifest['configuration']['computation']['block_svd_rtol'],
                    'MP_frobenius_norm_float64': validation['MP_frobenius_norm'],
                    'rank_interpretation': 'blockwise numerical rank; not exact algebraic rank'}
        return cls(torch.tensor(loaded, device=device), op.matrix, coordinates=coordinates,
                   image_weight=(2/radon.geometry.size)**2, contract=contract)

    def get_extra_state(self):
        return copy.deepcopy(self.contract)

    def set_extra_state(self, state):
        if state != self.contract:
            raise ValueError('Checkpoint projection contract mismatch')

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Enforce provenance even when a caller sets strict=False.
        self.set_extra_state(state_dict.get(prefix+'_extra_state'))
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def _apply(self, fn, recurse=True):
        probe = fn(torch.empty(0, dtype=torch.float64, device=self.basis.device))
        if probe.dtype != torch.float64:
            raise ValueError('Projection precision is fixed at Float64; conversion requires a new validation')
        return super()._apply(fn, recurse=recurse)

    def columns(self, value):
        if value.ndim not in (2, 3, 4) or value.shape[0] == 0:
            raise ValueError('Require a nonempty batch of vectors or single-channel images')
        if value.ndim == 4 and value.shape[1] != 1:
            raise ValueError('Only one image channel is supported')
        if math.prod(value.shape[1:]) != self.basis.shape[0]:
            raise ValueError('Image dimensions do not match projection')
        size = self.contract.get('geometry', {}).get('size')
        if size is not None and value.ndim >= 3 and tuple(value.shape[-2:]) != (size, size):
            raise ValueError('Image axes do not match the recorded geometry')
        if value.dtype != torch.float64 or value.device != self.basis.device:
            raise ValueError('Image and fixed operator must share Float64 dtype and device')
        if not bool(torch.isfinite(value).all()):
            raise ValueError('Nonfinite image/correction')
        return value.reshape(value.shape[0], -1).T

    def components(self, value):
        columns = self.columns(value)
        with torch.autocast(device_type=value.device.type, enabled=False):
            range_part = (self.basis @ (self.basis.T @ columns)).T.reshape_as(value)
            null_part = value-range_part
        return null_part, range_part

    def forward(self, value):
        return self.components(value)[0]

    def data(self, value):
        """Batch-first data in Euclidean/weighted-data coordinates, per contract."""
        columns = self.columns(value)
        with torch.autocast(device_type=value.device.type, enabled=False):
            product = torch.sparse.mm(self.matrix, columns) if self.matrix.is_sparse else self.matrix@columns
            return self.data_scale*product.T


def scalar_control(data_norm, beta):
    """Per-sample radial clipping; beta is a fixed nonnegative external budget.

At beta=0 choose the exact NSN branch, including at zero correction. At the
positive clipping kink use the unclipped branch derivative (not differentiable
there). No epsilon denominator or batch-global norm changes the radius.
"""
    if not bool(torch.isfinite(data_norm).all()) or bool((data_norm < 0).any()):
        raise ValueError('Data norms must be finite and nonnegative')
    if isinstance(beta, torch.Tensor) and beta.requires_grad:
        raise ValueError('Beta is an external budget, not a trainable parameter')
    budget = torch.as_tensor(beta, dtype=data_norm.dtype, device=data_norm.device)
    if budget.ndim == 0:
        budget = budget.expand_as(data_norm)
    if budget.shape != data_norm.shape or not bool(torch.isfinite(budget).all()) or bool((budget < 0).any()):
        raise ValueError('Beta must be a finite nonnegative scalar or one value per sample')
    active = (budget > 0) & (data_norm > budget)
    denominator = torch.where(active, data_norm, torch.ones_like(data_norm))
    ratio = budget/denominator
    control = torch.where(active, ratio, torch.ones_like(data_norm))
    return torch.where(budget == 0, torch.zeros_like(control), control), budget


class NullSpaceNetwork(nn.Module):
    def __init__(self, correction, projection):
        super().__init__()
        # Projection first: precision/contract guards execute before model casts/loads.
        self.projection = projection
        self.correction = correction

    def forward(self, z, *, return_details=False):
        self.projection.columns(z)
        value = self.correction(z)
        if value.shape != z.shape:
            raise ValueError('Correction network must preserve the exact input shape')
        null=self.projection(value)
        output=z+null
        if return_details:return output, {'value':value,'null':null,'range':value-null,'control':0.}
        return output


class ScalarDataProximalNetwork(NullSpaceNetwork):
    def forward(self, z, beta, *, return_details=False):
        self.projection.columns(z)
        value = self.correction(z)
        if value.shape != z.shape:
            raise ValueError('Correction network must preserve the exact input shape')
        null, range_part = self.projection.components(value)
        data = self.projection.data(value)
        norm = torch.linalg.vector_norm(data, dim=1)
        control, budget = scalar_control(norm, beta)
        scale = control.reshape((-1,)+(1,)*(z.ndim-1))
        output = z+null+scale*range_part
        if not return_details:
            return output
        # No detached tensors: diagnostics can inspect the actual derivative.
        return output, {'value': value, 'null': null, 'range': range_part,
                        'data_norm': norm, 'control': control, 'beta': budget}


class SeparateDataProximalNetwork(NullSpaceNetwork):
    """Independent U/V; clip the actual projected V data, bound beta+||A P U||.

The fixed numerical P need not be an exact null projector for this bound.
Both trainable branches are included in correction.parameters()/state_dict.
"""
    def __init__(self, null_correction, range_correction, projection):
        if {id(p) for p in null_correction.parameters()} & {id(p) for p in range_correction.parameters()}:
            raise ValueError('Separate branches must not share trainable parameters')
        super().__init__(nn.ModuleDict({'null':null_correction,'range':range_correction}),projection)

    def forward(self,z,beta,*,return_details=False):
        self.projection.columns(z)
        u=self.correction['null'](z);v=self.correction['range'](z)
        if u.shape!=z.shape or v.shape!=z.shape:
            raise ValueError('Both correction networks must preserve input shape')
        null=self.projection(u);_,ran=self.projection.components(v)
        norm=self.projection.data(ran).norm(dim=1)
        control,budget=scalar_control(norm,beta)
        output=z+null+control.reshape((-1,)+(1,)*(z.ndim-1))*ran
        if not return_details:return output
        return output,{'value':v,'null_value':u,'null':null,'range':ran,
                       'data_norm':norm,'control':control,'beta':budget}
