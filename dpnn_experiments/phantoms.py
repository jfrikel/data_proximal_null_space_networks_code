# SPDX-License-Identifier: GPL-3.0-only
"""Continuous ellipse objects, pixel area means and independent analytic sinograms."""

import hashlib
import math
import numpy as np


def rng_for(seed, object_id, purpose):
    message = f"{seed}:{object_id}:{purpose}".encode("utf-8")
    derived = int.from_bytes(hashlib.sha256(message).digest()[:16], "big")
    return np.random.Generator(np.random.PCG64(derived))


def object_ids(config):
    dataset = config["dataset"]
    namespace = dataset["namespace"]
    return ([f"{namespace}/circle"]
            + [f"{namespace}/oriented-{i:03d}" for i in range(len(dataset["probe_rotations_deg"]))]
            + [f"{namespace}/random-{i:03d}" for i in range(dataset["random_count"])])


def make_specimens(config):
    dataset = config["dataset"]
    namespace = dataset["namespace"]
    circle = dataset["circle"]
    specimens = [{"id": f"{namespace}/circle", "family": "circle", "ellipses": [circle]}]
    for i, angle in enumerate(dataset["probe_rotations_deg"]):
        ellipse = dict(dataset["probe_ellipse"], rotation_deg=angle)
        specimens.append({"id": f"{namespace}/oriented-{i:03d}", "family": "oriented_ellipse", "ellipses": [ellipse]})
    params = dataset["random_ellipses"]
    for i in range(dataset["random_count"]):
        name = f"{namespace}/random-{i:03d}"
        rng = rng_for(config["randomness"]["seed"], name, "geometry")
        count = int(rng.integers(params["count_min"], params["count_max"] + 1))
        amplitudes = rng.dirichlet(np.full(count, params["dirichlet_alpha"])) * rng.uniform(*params["total_amplitude_range"])
        ellipses = []
        for amplitude in amplitudes:
            for _ in range(params["max_rejection_attempts"]):
                p = rng.uniform(*params["major_axis_range"])
                q = p * rng.uniform(*params["axis_ratio_range"])
                center = rng.uniform(-params["support_radius"], params["support_radius"], 2)
                if np.linalg.norm(center) + p <= params["support_radius"]:
                    break
            else:
                raise ValueError("Phantom rejection limit exceeded; review parameter intervals")
            ellipses.append({"center": center.tolist(), "axes": [float(p), float(q)],
                             "rotation_deg": float(rng.uniform(0, 180)), "amplitude": float(amplitude)})
        specimens.append({"id": name, "family": "random_ellipses", "ellipses": ellipses})
    return specimens


def pixel_means(specimen, size, quadrature):
    if quadrature < 1 or size < 1:
        raise ValueError("Positive image and quadrature sizes required")
    result = np.zeros((size, size), dtype=np.float64)
    offsets = (np.arange(quadrature) + 0.5) / quadrature
    for ellipse in specimen["ellipses"]:
        p, q = ellipse["axes"]
        if min(p, q) <= 0:
            raise ValueError("Ellipse axes must be positive")
        c = ellipse["center"]
        phi = np.deg2rad(ellipse["rotation_deg"])
        co, si = np.cos(phi), np.sin(phi)
        for oy in offsets:
            yy = (-1 + (np.arange(size) + oy) * 2 / size - c[1])[:, None]
            for ox in offsets:
                xx = (-1 + (np.arange(size) + ox) * 2 / size - c[0])[None, :]
                inside = ((co * xx + si * yy) / p) ** 2 + ((-si * xx + co * yy) / q) ** 2 <= 1
                result += ellipse["amplitude"] / quadrature ** 2 * inside
    return result


def analytic_sinogram(specimen, theta, positions):
    theta = np.asarray(theta)[:, None]
    positions = np.asarray(positions)[None, :]
    result = np.zeros((theta.size, positions.size))
    for ellipse in specimen["ellipses"]:
        p, q = ellipse["axes"]
        phi = np.deg2rad(ellipse["rotation_deg"])
        d = np.sqrt((p * np.cos(theta - phi)) ** 2 + (q * np.sin(theta - phi)) ** 2)
        shift = ellipse["center"][0] * np.cos(theta) + ellipse["center"][1] * np.sin(theta)
        result += 2 * ellipse["amplitude"] * p * q / d * np.sqrt(np.maximum(0, 1 - ((positions - shift) / d) ** 2))
    return result


def boundary_samples(specimen, geometry, points, margin_deg):
    """Continuous normals modulo pi; these labels are not recovered wavefront sets."""
    if not (geometry.angle_min_deg == -60 and geometry.angle_max_deg == 60):
        raise ValueError("Direction-label convention currently supports only [-60,60) degrees")
    parameter = np.linspace(0, 2 * math.pi, points, endpoint=False)
    xy_parts, normal_parts, label_parts = [], [], []
    for ellipse in specimen["ellipses"]:
        p, q = ellipse["axes"]
        phi = np.deg2rad(ellipse["rotation_deg"])
        rotation = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
        xy = (rotation @ np.array([p * np.cos(parameter), q * np.sin(parameter)])).T + ellipse["center"]
        normals = (rotation @ np.array([np.cos(parameter) / p, np.sin(parameter) / q])).T
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        degrees = (np.rad2deg(np.arctan2(normals[:, 1], normals[:, 0])) + 90) % 180 - 90
        labels = np.ones(points, dtype=np.int8)  # 0 interior, 1 boundary band, 2 exterior
        labels[np.abs(degrees) < 60 - margin_deg] = 0
        labels[np.abs(degrees) > 60 + margin_deg] = 2
        xy_parts.append(xy)
        normal_parts.append(normals)
        label_parts.append(labels)
    return np.concatenate(xy_parts), np.concatenate(normal_parts), np.concatenate(label_parts)
