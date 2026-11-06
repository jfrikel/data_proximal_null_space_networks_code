# SPDX-License-Identifier: GPL-3.0-only
"""Pixelwise-constant line integrals with an exact algebraic transpose.

Images use x increasing along columns, y increasing along rows (plot origin=lower).
The domain is the full square, with no implicit circle mask. No ASTRA dependency.
"""

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class Geometry:
    size: int
    angles: int
    detectors: int
    angle_min_deg: float = -60.0
    angle_max_deg: float = 60.0
    half_width: float = math.sqrt(2)

    def __post_init__(self):
        if any(not isinstance(x, int) or isinstance(x, bool) for x in (self.size, self.angles, self.detectors)):
            raise ValueError("Geometry dimensions must be integers")
        if min(self.size, self.angles, self.detectors) < 1:
            raise ValueError("Geometry dimensions must be positive")
        if not 0 < self.angle_max_deg - self.angle_min_deg <= 180:
            raise ValueError("Angular interval must have width in (0,180]")
        if not math.isfinite(self.half_width) or self.half_width < math.sqrt(2):
            raise ValueError("Full-square model requires detector coverage [-sqrt(2),sqrt(2)]")

    @property
    def theta(self):
        return np.linspace(np.deg2rad(self.angle_min_deg), np.deg2rad(self.angle_max_deg), self.angles, endpoint=False)

    @property
    def positions(self):
        ds = 2 * self.half_width / self.detectors
        return -self.half_width + (np.arange(self.detectors) + 0.5) * ds

    @property
    def data_weight(self):
        return np.deg2rad(self.angle_max_deg - self.angle_min_deg) / self.angles * 2 * self.half_width / self.detectors

    def data_norm(self, value):
        return float(np.linalg.norm(value) * math.sqrt(self.data_weight))


class SparseRadon:
    def __init__(self, geometry, rows, cols, values):
        self.geometry = geometry
        self.rows = np.asarray(rows, dtype=np.int32)
        self.cols = np.asarray(cols, dtype=np.int32)
        self.values = np.asarray(values, dtype=np.float64)
        self.shape = (geometry.angles * geometry.detectors, geometry.size ** 2)

    def forward(self, image):
        image = np.asarray(image, dtype=np.float64).reshape(-1)
        if image.size != self.shape[1]:
            raise ValueError("Incorrect image size")
        return np.bincount(self.rows, weights=self.values * image[self.cols], minlength=self.shape[0])

    def transpose(self, data):
        data = np.asarray(data, dtype=np.float64).reshape(-1)
        if data.size != self.shape[0]:
            raise ValueError("Incorrect data size")
        return np.bincount(self.cols, weights=self.values * data[self.rows], minlength=self.shape[1])

    def adjoint(self, data):
        # Physical L2 pixel and measurement quadrature norms.
        return self.transpose(data) * self.geometry.data_weight / (2 / self.geometry.size) ** 2

    def dense(self):
        if self.shape[0] * self.shape[1] > 20_000_000:
            raise ValueError("Dense conversion restricted to small reference operators")
        matrix = np.zeros(self.shape)
        np.add.at(matrix, (self.rows, self.cols), self.values)
        return matrix

    def save(self, path):
        np.savez_compressed(path, rows=self.rows, cols=self.cols, values=self.values, shape=self.shape)


def build_radon(geometry, progress=None, ray_tolerance=1e-12):
    """Intersect each ray with all grid lines; integrate constant pixels exactly."""
    n = geometry.size
    edges = np.linspace(-1.0, 1.0, n + 1)
    row_chunks, col_chunks, value_chunks = [], [], []
    eps = float(ray_tolerance)
    if not 0 < eps < 1e-6:
        raise ValueError("Ray tolerance must be in (0,1e-6)")
    for ai, theta in enumerate(geometry.theta):
        normal = np.array([math.cos(theta), math.sin(theta)])
        direction = np.array([-math.sin(theta), math.cos(theta)])
        for si, position in enumerate(geometry.positions):
            origin = position * normal
            lo, hi = -math.inf, math.inf
            cuts = []
            for axis in range(2):
                if abs(direction[axis]) < eps:
                    if abs(origin[axis]) > 1:
                        hi = -math.inf
                        break
                else:
                    ts = (edges - origin[axis]) / direction[axis]
                    lo = max(lo, min(ts[0], ts[-1]))
                    hi = min(hi, max(ts[0], ts[-1]))
                    cuts.extend(ts[1:-1])
            if not math.isfinite(lo) or not math.isfinite(hi) or hi - lo <= eps:
                continue
            ts = np.array(cuts)
            ts = np.sort(np.r_[lo, ts[(ts > lo + eps) & (ts < hi - eps)], hi])
            lengths = np.diff(ts)
            good = lengths > eps
            midpoints = origin[:, None] + direction[:, None] * ((ts[:-1] + ts[1:]) / 2)[None, :]
            indices = np.clip(np.floor((midpoints + 1) * n / 2).astype(np.int64), 0, n - 1)
            cols = indices[1, good] * n + indices[0, good]
            row_chunks.append(np.full(cols.size, ai * geometry.detectors + si, dtype=np.int32))
            col_chunks.append(cols.astype(np.int32))
            value_chunks.append(lengths[good])
        if progress:
            progress(ai + 1, geometry.angles)
    if not value_chunks:
        return SparseRadon(geometry, [], [], [])
    return SparseRadon(geometry, np.concatenate(row_chunks), np.concatenate(col_chunks), np.concatenate(value_chunks))


def storage_estimate(geometry):
    m, n = geometry.angles * geometry.detectors, geometry.size ** 2
    r = min(m, n)
    return {"rows": m, "columns": n, "nullity_dimension_lower_bound": max(0, n - m),
            "dense_float64_gib": 8 * m * n / 2 ** 30,
            "dense_A_plus_economy_U_V_float64_gib": 8 * (m*n + m*r + n*r) / 2 ** 30,
            "excludes": "SVD workspace, copies, network, gradients and batches"}
