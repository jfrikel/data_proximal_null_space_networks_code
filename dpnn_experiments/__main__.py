"""Run with python -m dpnn_experiments {check,prepare,verify}."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import resource
import sys
import time
import unittest

import numpy as np

from dpnn_experiments.operators import Geometry, build_radon, storage_estimate
from dpnn_experiments.phantoms import analytic_sinogram, boundary_samples, make_specimens, object_ids, pixel_means, rng_for
from dpnn_experiments.provenance import RunArchive, digest, write_json


SOURCE_ROOT = Path(__file__).resolve().parents[1]


def read_config(path):
    value = json.loads(Path(path).read_text())
    if "configuration" in value:
        # Replay must use the recorded source, including uncommitted files.
        for entry in value["source_files"]:
            candidate = SOURCE_ROOT / entry["path"]
            if not candidate.is_file() or digest(candidate) != entry["sha256"]:
                raise ValueError("Replay source mismatch; extract sources.zip and execute that snapshot")
        parent_run_id = value["run_id"]
        value = value["configuration"]
        value["replay_of_run_id"] = parent_run_id
    computation = value["computation"]
    supported = {"backend": "numpy_cpu", "dtype": "float64", "operator": "pixel_line_intersections_v1",
                 "detector_measurement": "central line integral", "active_pixel_mask": "full_square"}
    for key, expected in supported.items():
        if computation.get(key) != expected:
            raise ValueError(f"This preparation entry point requires {key}={expected}")
    if value["dataset"]["split"] != "development_control":
        raise ValueError("This diagnostic generator must not create main training/validation/test data")
    geometry = Geometry(**value["geometry"])
    if (geometry.angle_min_deg, geometry.angle_max_deg) != (-60, 60):
        raise ValueError("Direction diagnostics currently require [-60,60) degrees")
    if computation["profile_repetitions"] < 1 or computation["relative_noise_norm"] < 0:
        raise ValueError("Invalid profiling/noise parameters")
    if not 0 < computation["direction_margin_deg"] < 30:
        raise ValueError("Direction margin must be between 0 and 30 degrees")
    sizes = computation["refinement_sizes"]
    if geometry.size not in sizes or len(set(sizes)) != len(sizes) or any(not isinstance(n, int) or n < 1 for n in sizes):
        raise ValueError("Refinement sizes must be unique positive integers and include target size")
    for key in ("pixel_quadrature", "comparison_quadrature", "boundary_points_per_ellipse", "profile_repetitions"):
        if not isinstance(computation[key], int) or computation[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    declarations = {"pixel_domain": [-1.0, 1.0], "image_orientation": "rows increase with y; columns increase with x",
                    "measurement_order": "angle-major; detector-minor", "angle_endpoint": False,
                    "pixel_norm_weight": "(2/size)^2", "data_norm_weight": "angular_step * detector_step",
                    "noise_model": "normalized independent Gaussian direction, fixed weighted L2 norm"}
    for key, expected in declarations.items():
        if computation[key] != expected:
            raise ValueError(f"Unsupported numerical convention: {key}")
    value["dataset"]["split_ids"] = {"development_control": object_ids(value)}
    if len(set(object_ids(value))) != len(object_ids(value)):
        raise ValueError("Duplicate specimen IDs")
    return value


def profile_operator(operator, archive, config):
    rng = rng_for(config["randomness"]["seed"], str(operator.geometry.size), "operator_profile")
    x = rng.normal(size=operator.shape[1])
    y = rng.normal(size=operator.shape[0])
    lhs = float(np.dot(operator.forward(x), y) * operator.geometry.data_weight)
    rhs = float(np.dot(x, operator.adjoint(y)) * (2 / operator.geometry.size) ** 2)
    defect = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1.0)
    timings = {"forward_seconds": [], "transpose_seconds": []}
    for _ in range(config["computation"]["profile_repetitions"]):
        start = time.perf_counter()
        operator.forward(x)
        timings["forward_seconds"].append(time.perf_counter() - start)
        start = time.perf_counter()
        operator.transpose(y)
        timings["transpose_seconds"].append(time.perf_counter() - start)
    result = {"geometry": asdict(operator.geometry), "nnz": int(operator.values.size),
              "coo_array_bytes": operator.rows.nbytes + operator.cols.nbytes + operator.values.nbytes,
              "weighted_adjoint_relative_defect": defect, **storage_estimate(operator.geometry), **timings}
    archive.event("operator_profile", **result)
    if defect > config["computation"]["adjoint_tolerance"]:
        raise ArithmeticError(f"Adjoint test failed: {defect}")
    return result


def prepare(archive, config):
    geometry = Geometry(**config["geometry"])
    options = config["computation"]
    specimens = make_specimens(config)
    write_json(archive.path / "specimens.json", specimens)
    write_json(archive.path / "splits.json", config["dataset"]["split_ids"])
    metrics, profiles = [], []
    # Measurement positions and continuous objects stay fixed during refinement.
    for size in options["refinement_sizes"]:
        current = replace(geometry, size=size)
        archive.event("operator_build_started", geometry=asdict(current))
        start = time.perf_counter()
        operator = build_radon(current, ray_tolerance=options["ray_tolerance"], progress=lambda done, total:
                               archive.event("operator_build_progress", size=size, angles_done=done, total=total)
                               if done % 20 == 0 or done == total else None)
        build_seconds = time.perf_counter() - start
        profile = profile_operator(operator, archive, config)
        profile["build_seconds"] = build_seconds
        profiles.append(profile)
        print(f"{size} x {size}: {operator.values.size:,} nonzero matrix entries", flush=True)
        if size == geometry.size:
            operator.save(archive.path / "operator.npz")
        for specimen in specimens:
            name = specimen["id"].split("/")[-1]
            archive.event("specimen_started", object_id=specimen["id"], image_size=size)
            image = pixel_means(specimen, size, options["pixel_quadrature"])
            comparison = pixel_means(specimen, size, options["comparison_quadrature"])
            exact = analytic_sinogram(specimen, current.theta, current.positions).ravel()
            matched = operator.forward(image)
            random = rng_for(config["randomness"]["seed"], specimen["id"], "noise")
            noise_state = random.bit_generator.state
            direction = random.normal(size=exact.size)
            direction /= current.data_norm(direction)
            delta = options["relative_noise_norm"] * current.data_norm(exact)
            delta_matched = options["relative_noise_norm"] * current.data_norm(matched)
            error = current.data_norm(exact - matched)
            quadrature_error = float(np.linalg.norm(image - comparison) * 2 / size)
            projected_quadrature_difference = current.data_norm(operator.forward(image - comparison))
            record = {"object_id": specimen["id"], "family": specimen["family"], "split": "development_control",
                      "size": size, "model_error_norm": error,
                      "relative_model_error": error / max(current.data_norm(exact), 1e-30),
                      "pixel_quadrature_difference_norm": quadrature_error,
                      "projected_quadrature_difference_norm": projected_quadrature_difference,
                      "relative_projected_quadrature_difference": projected_quadrature_difference / max(current.data_norm(exact), 1e-30),
                      "noise_norm_analytic": delta, "noise_norm_matched": delta_matched,
                      "model_error_to_noise_ratio": error / delta if delta else None,
                      "pixel_min": float(image.min()), "pixel_max": float(image.max())}
            metrics.append(record)
            archive.event("specimen_evaluated", **record)
            directory = archive.path / "data" / str(size)
            directory.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(directory / f"{name}.npz", image=image, analytic=exact, matched=matched,
                                noisy_analytic=exact + delta * direction, noisy_matched=matched + delta_matched * direction)
            if size == geometry.size:
                xy, normals, labels = boundary_samples(specimen, current, options["boundary_points_per_ellipse"], options["direction_margin_deg"])
                np.savez_compressed(directory / f"{name}_boundaries.npz", xy=xy, normals=normals, labels=labels)
                write_json(directory / f"{name}_metadata.json", {"object_id": specimen["id"], "noise_rng_initial_state": noise_state,
                    "boundary_label_counts": {str(i): int((labels == i).sum()) for i in range(3)},
                    "boundary_labels": {"0": "measured-angle interior", "1": "angular boundary band", "2": "outside measured angles"},
                    "qualification": "Continuous geometric normals; no measured microlocal guarantee or edge-recovery score."})
        del operator
    write_json(archive.path / "per_object_metrics.json", metrics)
    write_json(archive.path / "operator_profiles.json", profiles)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write_json(archive.path / "summary.json", {
        "purpose": config["purpose"], "objects": len(specimens), "split": "development_control",
        "resolutions": options["refinement_sizes"], "backend": "CPU NumPy; no H200 benchmark",
        "peak_process_rss_bytes": peak if sys.platform == "darwin" else peak * 1024,
        "by_resolution": {str(size): {"max_relative_model_error": max(x["relative_model_error"] for x in metrics if x["size"] == size),
                                      "mean_relative_model_error": float(np.mean([x["relative_model_error"] for x in metrics if x["size"] == size]))}
                          for size in options["refinement_sizes"]},
        "not_performed": ["training", "main test evaluation", "large SVD", "iterative GPU projection", "microlocal reconstruction metrics"],
    })


def check(archive):
    from tests import test_preparation
    test_preparation.ARCHIVE = archive

    class LoggedResult(unittest.TextTestResult):
        def startTest(self, test):
            archive.event("test_started", test_id=test.id())
            super().startTest(test)

        def addSuccess(self, test):
            archive.event("test_passed", test_id=test.id())
            super().addSuccess(test)

        def addFailure(self, test, err):
            archive.event("test_failed", test_id=test.id(), error=str(err[1]))
            super().addFailure(test, err)

        def addError(self, test, err):
            archive.event("test_error", test_id=test.id(), error=str(err[1]))
            super().addError(test, err)

    result = unittest.TextTestRunner(verbosity=2, resultclass=LoggedResult).run(unittest.defaultTestLoader.loadTestsFromModule(test_preparation))
    write_json(archive.path / "test_results.json", {"tests": result.testsRun, "failures": len(result.failures), "errors": len(result.errors)})
    if not result.wasSuccessful():
        raise RuntimeError("Preparation checks failed; see archived test output")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "check", "verify"])
    parser.add_argument("--config", type=Path, default=SOURCE_ROOT / "configs" / "preparation" / "prepare_256.json")
    parser.add_argument("--runs-root", type=Path, default=SOURCE_ROOT / "runs")
    parser.add_argument("--mirror-root", type=Path, default=None)
    parser.add_argument("--archive", type=Path, help="Existing run directory to verify")
    args = parser.parse_args()
    if args.action == "verify":
        if args.archive is None:
            parser.error("verify requires --archive")
        records = json.loads((args.archive / "artifacts.json").read_text())["files"]
        failed = [entry["path"] for entry in records if not (args.archive / entry["path"]).is_file()
                  or digest(args.archive / entry["path"]) != entry["sha256"]]
        if failed:
            raise RuntimeError(f"Artifact verification failed: {failed}")
        print(f"Verified {len(records)} artifact hashes.")
        return
    config = read_config(args.config)
    config["execution_action"] = args.action
    with RunArchive(config, args.runs_root, SOURCE_ROOT, args.mirror_root, parent_run_id=config.get("replay_of_run_id")) as archive:
        if args.action == "check":
            check(archive)
        else:
            prepare(archive, config)
    print(f"Run archive: {archive.path}")
    print(f"Metadata mirror: {archive.mirror}")


if __name__ == "__main__":
    main()
