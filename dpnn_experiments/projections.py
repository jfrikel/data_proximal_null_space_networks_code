# SPDX-License-Identifier: GPL-3.0-only
"""Fixed-operator inversion in Euclidean (whitened) coordinates.

For the Radon geometry use M=sqrt(w_Y/w_X)*A, x~=sqrt(w_X)*x,
y~=sqrt(w_Y)*y. The physical inverse is B(y)=M^dagger(y~)/sqrt(w_X).
Adaptive LSQR is an evaluation solver, not a differentiable projector. The
fixed Landweber polynomials have their actual finite-iteration adjoints.
"""

import math
import numpy as np
import torch


class MatrixOperator:
    def __init__(self, matrix, norm_bound_squared):
        if matrix.requires_grad or matrix.dtype != torch.float64:
            raise ValueError("Require a fixed Float64 matrix")
        if not math.isfinite(norm_bound_squared) or norm_bound_squared < 0:
            raise ValueError("Invalid squared norm bound")
        self.matrix = matrix
        self.sparse = matrix.layout == torch.sparse_coo
        self.transposed = matrix.transpose(0, 1).coalesce() if self.sparse else matrix.T.contiguous()
        self.shape = tuple(matrix.shape)
        self.norm_bound_squared = norm_bound_squared

    @classmethod
    def dense(cls, array, device):
        array = np.asarray(array, dtype=np.float64)
        bound = float(np.abs(array).sum(axis=0).max() * np.abs(array).sum(axis=1).max())
        return cls(torch.tensor(array, dtype=torch.float64, device=device), bound)

    @classmethod
    def whitened_radon(cls, operator, device):
        factor = math.sqrt(operator.geometry.data_weight) / (2 / operator.geometry.size)
        values = operator.values * factor
        norm_inf = np.bincount(operator.rows, weights=np.abs(values), minlength=operator.shape[0]).max()
        norm_one = np.bincount(operator.cols, weights=np.abs(values), minlength=operator.shape[1]).max()
        matrix = torch.sparse_coo_tensor(
            torch.from_numpy(np.stack((operator.rows, operator.cols)).astype(np.int64)),
            torch.from_numpy(values), operator.shape).coalesce().to(device)
        return cls(matrix, float(norm_inf * norm_one))

    def _multiply(self, matrix, value):
        if value.ndim not in (1, 2) or value.shape[0] != matrix.shape[1]:
            raise ValueError("Expected a vector or column batch of the correct size")
        if self.sparse:
            result = torch.sparse.mm(matrix, value[:, None] if value.ndim == 1 else value)
            return result[:, 0] if value.ndim == 1 else result
        return matrix @ value

    def forward(self, value):
        return self._multiply(self.matrix, value)

    def transpose(self, value):
        return self._multiply(self.transposed, value)


def null_polynomial_native(value, operator, step, iterations):
    result = value
    for _ in range(iterations):
        result = result - step * operator.transpose(operator.forward(result))
    return result


def inverse_polynomial_native(data, operator, step, iterations):
    shape = (operator.shape[1],) + tuple(data.shape[1:])
    result = data.new_zeros(shape)
    for _ in range(iterations):
        result = result + step * operator.transpose(data - operator.forward(result))
    return result


class _NullPolynomial(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, operator, step, iterations):
        ctx.constants = (operator, step, iterations)
        return null_polynomial_native(value, operator, step, iterations)

    @staticmethod
    def backward(ctx, gradient):
        # This is the adjoint of P_K, not the exact kernel projector P.
        return _NullPolynomial.apply(gradient, *ctx.constants), None, None, None


class _InversePolynomial(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data, operator, step, iterations):
        ctx.constants = (operator, step, iterations)
        return inverse_polynomial_native(data, operator, step, iterations)

    @staticmethod
    def backward(ctx, gradient):
        operator, step, iterations = ctx.constants
        shape = (operator.shape[0],) + tuple(gradient.shape[1:])
        result = gradient.new_zeros(shape)
        rhs = operator.forward(gradient)
        for _ in range(iterations):
            result = result + step * (rhs - operator.forward(operator.transpose(result)))
        return result, None, None, None


class FixedLandweber:
    """B_K approximates M^dagger; P_K=I-B_K M generally is not idempotent.

The matrix, step and iteration count are fixed. No gradients with respect to
them are supported. Vector inputs and independent column batches are accepted.
"""
    def __init__(self, operator, iterations, step=None):
        if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 0:
            raise ValueError("Iterations must be a nonnegative integer")
        bound = operator.norm_bound_squared
        step = (0.95 / bound if bound else 1.0) if step is None else float(step)
        if not math.isfinite(step) or step <= 0 or step * bound > 1:
            raise ValueError("Require 0 < step <= 1 / norm_bound_squared")
        self.operator, self.iterations, self.step = operator, iterations, step

    def null(self, value):
        return _NullPolynomial.apply(value, self.operator, self.step, self.iterations)

    def inverse(self, data):
        return _InversePolynomial.apply(data, self.operator, self.step, self.iterations)


def lsqr(operator, data, *, max_iterations, normal_rtol, data_rtol,
         require_data_residual, check_interval, roundoff_factor, condition_limit, callback=None):
    """Zero-start, undamped, unpreconditioned Golub-Kahan LSQR.

Stopping uses explicitly recomputed residuals. No claim about image error
follows from those residuals. The entire iteration history is returned; a
callback receives each row and the current solution. Autograd is rejected
because adaptive LSQR is not a fixed linear map at finite tolerance.
"""
    if data.requires_grad:
        raise ValueError("Adaptive LSQR has no autograd contract; use FixedLandweber")
    if data.ndim != 1 or data.shape[0] != operator.shape[0]:
        raise ValueError("LSQR requires one measurement vector")
    if (isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 1
            or not isinstance(check_interval, int) or check_interval < 1
            or not math.isfinite(normal_rtol) or not 0 < normal_rtol < 1
            or not math.isfinite(data_rtol) or not 0 < data_rtol < 1
            or not math.isfinite(roundoff_factor) or roundoff_factor < 0
            or not math.isfinite(condition_limit) or condition_limit <= 1):
        raise ValueError("Invalid LSQR iteration/stopping parameters")
    with torch.no_grad():
        norm = lambda value: float(torch.linalg.vector_norm(value).item())
        result = data.new_zeros(operator.shape[1])
        data_scale = norm(data)
        normal_scale = norm(operator.transpose(data))
        history = []
        norm_bound = math.sqrt(operator.norm_bound_squared)
        inverse_norm_squared = 0.0
        condition_estimate = 0.0

        def thresholds():
            solution_norm = norm(result)
            rounding = roundoff_factor * torch.finfo(data.dtype).eps * (data_scale + norm_bound * solution_norm)
            return (normal_rtol * normal_scale + norm_bound * rounding,
                    data_rtol * data_scale + rounding, solution_norm)

        def record(k, estimate, normal_estimate, alpha, beta, true_check):
            row = {"iteration": k, "estimated_residual_norm": estimate,
                   "estimated_normal_residual_norm": normal_estimate, "alpha": alpha, "beta": beta,
                   "condition_estimate": condition_estimate}
            if true_check:
                normal_threshold, data_threshold, solution_norm = thresholds()
                residual = data - operator.forward(result)
                row.update(residual_norm=norm(residual), normal_residual_norm=norm(operator.transpose(residual)),
                           normal_threshold=normal_threshold, data_threshold=data_threshold, solution_norm=solution_norm)
                row["normal_stop"] = row["normal_residual_norm"] <= normal_threshold
                row["data_stop"] = row["residual_norm"] <= data_threshold
            if any(isinstance(x, float) and not math.isfinite(x) for x in row.values()):
                raise ArithmeticError("Nonfinite LSQR state")
            history.append(row)
            if callback is not None:
                callback(row, result)
            return row

        record(0, data_scale, normal_scale, 0.0, data_scale, True)
        if data_scale == 0 or normal_scale == 0:
            return result, {"stop_reason": "stationary_at_zero", "iterations": 0,
                            "converged": not require_data_residual or data_scale == 0,
                            "data_scale": data_scale, "normal_scale": normal_scale, "history": history}
        beta = data_scale
        u = data / beta
        v = operator.transpose(u)
        alpha = norm(v)
        v = v / alpha
        w = v.clone()
        rho_bar, phi_bar = alpha, beta
        reason, converged = "iteration_limit", False
        for k in range(1, max_iterations + 1):
            u = operator.forward(v) - alpha * u
            beta = norm(u)
            if beta > 0:
                u = u / beta
            v = operator.transpose(u) - beta * v
            alpha = norm(v)
            if alpha > 0:
                v = v / alpha
            rho = math.hypot(rho_bar, beta)
            if rho == 0:
                reason = "zero_rotation_breakdown"
                record(k, abs(phi_bar), None, alpha, beta, True)
                break
            cosine, sine = rho_bar / rho, beta / rho
            theta, rho_bar = sine * alpha, -cosine * alpha
            phi, phi_bar = cosine * phi_bar, sine * phi_bar
            inverse_norm_squared += (norm(w) / rho) ** 2
            condition_estimate = norm_bound * math.sqrt(inverse_norm_squared)
            result = result + (phi / rho) * w
            w = v - (theta / rho) * w
            terminal = alpha == 0 or beta == 0
            normal_estimate = alpha * abs(sine * phi)
            # Do not step past a converged low-rank solution merely to reach the
            # next logging/check interval: roundoff can then amplify null modes.
            normal_threshold, data_threshold, _ = thresholds()
            candidate = normal_estimate <= normal_threshold and (
                abs(phi_bar) <= data_threshold or not require_data_residual)
            condition_stop = condition_estimate >= condition_limit
            row = record(k, abs(phi_bar), normal_estimate, alpha, beta,
                         candidate or condition_stop or terminal or k % check_interval == 0 or k == max_iterations)
            if row.get("normal_stop") and (row.get("data_stop") or not require_data_residual):
                reason, converged = "true_residual_tolerances", True
                break
            if condition_stop:
                reason = "condition_limit"
                break
            if terminal:
                reason = "bidiagonalization_terminated_before_tolerance"
                break
        return result, {"stop_reason": reason, "iterations": history[-1]["iteration"],
                        "converged": converged, "data_scale": data_scale,
                        "normal_scale": normal_scale, "history": history}
