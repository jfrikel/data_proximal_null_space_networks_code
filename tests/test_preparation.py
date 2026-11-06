# SPDX-License-Identifier: GPL-3.0-only
"""Scientific and archive checks; invoke through the logged 'check' command."""

import json
from pathlib import Path
import unittest

import numpy as np

from dpnn_experiments.operators import Geometry, build_radon
from dpnn_experiments.phantoms import analytic_sinogram, boundary_samples, make_specimens, pixel_means, rng_for
from dpnn_experiments.provenance import RunArchive, digest, write_json

ARCHIVE = None


class PreparationChecks(unittest.TestCase):
    def setUp(self):
        if ARCHIVE is None:
            raise RuntimeError("Run checks with python -m dpnn_experiments check to retain provenance")
        self.rng = np.random.Generator(np.random.PCG64(123))
        ARCHIVE.event("test_configuration", test_id=self.id(), seed=123, dtype="float64", ray_tolerance=1e-12)

    def geometry(self, **params):
        geometry = Geometry(**params)
        ARCHIVE.event("test_geometry", test_id=self.id(), parameters=params)
        return geometry

    def test_axis_and_diagonal_square_chords(self):
        geometry = self.geometry(size=8, angles=1, detectors=9, angle_min_deg=0, angle_max_deg=1)
        operator = build_radon(geometry)
        expected = np.where(np.abs(geometry.positions) < 1, 2.0, 0.0)
        np.testing.assert_allclose(operator.forward(np.ones(64)), expected, atol=1e-12)
        diagonal = self.geometry(size=9, angles=1, detectors=1, angle_min_deg=45, angle_max_deg=46)
        self.assertAlmostEqual(build_radon(diagonal).forward(np.ones(81))[0], 2*np.sqrt(2), places=12)

    def test_transpose_and_weighted_adjoint(self):
        geometry = self.geometry(size=16, angles=8, detectors=23)
        operator = build_radon(geometry)
        x = self.rng.normal(size=256)
        y = self.rng.normal(size=184)
        np.testing.assert_allclose(np.dot(operator.forward(x), y), np.dot(x, operator.transpose(y)), atol=1e-11)
        np.testing.assert_allclose(np.dot(operator.forward(x), y)*geometry.data_weight,
                                   np.dot(x, operator.adjoint(y))*(2/16)**2, atol=1e-12)

    def test_svd_nullspace_and_dp_bound(self):
        geometry = self.geometry(size=16, angles=8, detectors=23)
        matrix = build_radon(geometry).dense() * np.sqrt(geometry.data_weight)/(2/16)
        _, singular, vh = np.linalg.svd(matrix, full_matrices=False)
        threshold = 1e-12 * singular[0]
        basis = vh[singular > threshold]
        v = self.rng.normal(size=256)
        q = basis.T @ (basis @ v)
        n = v - q
        ARCHIVE.event("test_svd", rank=len(basis), threshold=float(threshold),
                      null_defect=float(np.linalg.norm(matrix@n)), beta_values=[0, 0.05, 100])
        self.assertLess(np.linalg.norm(matrix@n), 1e-11)
        self.assertLess(abs(np.dot(n,q)), 1e-10)
        np.testing.assert_allclose(n - basis.T@(basis@n), n, atol=1e-12)
        for beta in (0, 0.05, 100):
            t = min(1, beta/np.linalg.norm(matrix@v))
            self.assertLessEqual(np.linalg.norm(matrix@(n+t*q)), beta+1e-11)

    def test_approximate_projection_leak_bound(self):
        geometry = self.geometry(size=8, angles=4, detectors=12)
        matrix = build_radon(geometry).dense()
        v = self.rng.normal(size=64)
        n = v.copy()
        omega = 1/np.linalg.norm(matrix, 2)**2
        for _ in range(5):
            n -= omega * matrix.T@(matrix@n)
        beta = 0.03
        t = min(1, beta/np.linalg.norm(matrix@v))
        actual = np.linalg.norm(matrix@(n+t*(v-n)))
        bound = beta+(1-t)*np.linalg.norm(matrix@n)
        ARCHIVE.event("test_projection_leak", iterations=5, omega=float(omega), beta=beta, actual=float(actual), bound=float(bound))
        self.assertLessEqual(actual, bound+1e-12)

    def test_analytic_circle_and_mass(self):
        specimen = {"ellipses": [{"center": [0,0], "axes": [0.6,0.6], "rotation_deg": 37, "amplitude": 0.8}]}
        positions = np.linspace(-1,1,20001)
        theta = np.array([0, 0.2, 1.3])
        data = analytic_sinogram(specimen, theta, positions)
        expected = 1.6*np.sqrt(np.maximum(0,0.36-positions**2))
        np.testing.assert_allclose(data, np.broadcast_to(expected,data.shape), atol=1e-12)
        mass = np.sum((data[0,1:]+data[0,:-1])/2)* (positions[1]-positions[0])
        self.assertAlmostEqual(mass, np.pi*0.36*0.8, places=5)
        ARCHIVE.event("test_analytic_mass", specimen=specimen, detector_positions=20001, actual=float(mass))

    def test_phantom_identity_and_noise_budget(self):
        config = json.loads(json.dumps(ARCHIVE.config))
        first = make_specimens(config)
        config["geometry"]["size"] = 64
        self.assertEqual(first, make_specimens(config))
        geometry = self.geometry(size=8, angles=6, detectors=12)
        random = rng_for(41, "unit-object", "noise")
        direction = random.normal(size=72)
        eta = 0.07*direction/geometry.data_norm(direction)
        self.assertAlmostEqual(geometry.data_norm(eta), 0.07, places=14)
        image = pixel_means(first[-1], 32, 4)
        self.assertGreaterEqual(image.min(),0)
        self.assertLessEqual(image.max(),1+1e-12)
        ARCHIVE.event("test_phantom_parameters", specimens=first, noise_seed=41, noise_budget=0.07, raster_size=32, quadrature=4)

    def test_direction_labels(self):
        geometry = self.geometry(size=16, angles=8, detectors=23)
        specimen = {"ellipses": [{"center": [0,0], "axes": [0.5,0.5], "rotation_deg": 0, "amplitude": 1}]}
        _, normals, labels = boundary_samples(specimen, geometry, 12, 5)
        self.assertEqual(labels[0],0)   # x normal: measured.
        self.assertEqual(labels[2],1)   # 60 degrees: boundary.
        self.assertEqual(labels[3],2)   # y normal: outside measured interval.
        np.testing.assert_allclose(np.linalg.norm(normals,axis=1),1)
        ARCHIVE.event("test_boundary_parameters", specimen=specimen, boundary_points=12, margin_deg=5)

    def test_archive_failure_and_source_integrity(self):
        source_root = Path(__file__).resolve().parents[1]
        with self.assertRaisesRegex(RuntimeError, "intentional fixture failure"):
            with RunArchive(ARCHIVE.config, ARCHIVE.path/"fixtures", source_root, parent_run_id=ARCHIVE.run_id) as child:
                write_json(child.path/"fixture.json", {"purpose": "intentional logger failure test"})
                raise RuntimeError("intentional fixture failure")
        completion = json.loads((child.path/"completion.json").read_text())
        self.assertEqual(completion["status"], "failed")
        manifest = json.loads((child.path/"manifest.json").read_text())
        self.assertEqual(digest(child.path/"sources.zip"),manifest["sources_zip_sha256"])
        records = json.loads((child.path/"artifacts.json").read_text())["files"]
        self.assertTrue(all(digest(child.path/x["path"])==x["sha256"] for x in records))
        with self.assertRaisesRegex(ValueError,"Missing required"):
            RunArchive({}, ARCHIVE.path/"fixtures", source_root)

    def test_unwritable_mirror_prevents_computation(self):
        source_root = Path(__file__).resolve().parents[1]
        root = ARCHIVE.path/"fixtures"
        root.mkdir(exist_ok=True)
        blocked = root/"not-a-directory"
        blocked.write_text("intentional fixture")
        child = RunArchive(ARCHIVE.config, root, source_root, mirror_root=blocked, parent_run_id=ARCHIVE.run_id)
        reached = False
        with self.assertRaises(OSError):
            with child:
                reached = True
        self.assertFalse(reached)
        self.assertEqual(json.loads((child.path/"completion.json").read_text())["status"],"setup_failed")

    def test_gpu_reference_integrity_and_configuration(self):
        from dataclasses import asdict
        from dpnn_experiments.gpu_benchmark import load_operator, make_config, reference_metadata
        root = ARCHIVE.path / "fixtures" / "gpu-reference"
        root.mkdir(parents=True)
        geometry = self.geometry(size=8, angles=4, detectors=12)
        operator = build_radon(geometry)
        operator.save(root / "operator.npz")
        write_json(root / "manifest.json", {"run_id": "intentional-reference-fixture", "configuration": {
            "execution_action": "prepare", "geometry": asdict(geometry),
            "computation": {"operator": "pixel_line_intersections_v1", "dtype": "float64"}}})
        write_json(root / "completion.json", {"status": "completed"})
        files = [{"path": p.name, "sha256": digest(p)} for p in sorted(root.iterdir())]
        write_json(root / "artifacts.json", {"files": files})
        reference = reference_metadata(root)
        restored = load_operator(root, reference)
        np.testing.assert_array_equal(restored.values, operator.values)
        config = make_config(reference, 123, "cuda:0", 2)
        self.assertEqual(config["computation"]["backend"], "torch_cuda")
        with self.assertRaises(ValueError):
            make_config(reference, 123, "cpu", 2)
        with self.assertRaises(ValueError):
            make_config(reference, 123, "cuda:0", 0)
        with (root / "operator.npz").open("ab") as stream:
            stream.write(b"intentional corruption fixture")
        with self.assertRaisesRegex(ValueError, "integrity check failed"):
            reference_metadata(root)
        with self.assertRaisesRegex(ValueError, "Operator changed"):
            load_operator(root, reference)
