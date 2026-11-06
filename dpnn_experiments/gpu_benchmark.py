"""Logged CUDA checks of an archived Radon matrix, without training or inversion.

Run only on an allocated GPU or a server permitting direct GPU use. There is no
CPU fallback: unavailable CUDA or failed numerical checks produce failed runs.
"""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from dpnn_experiments.operators import Geometry, SparseRadon
from dpnn_experiments.provenance import RunArchive, digest, write_json

SOURCE_ROOT = Path(__file__).resolve().parents[1]


def reference_metadata(root):
    root = Path(root).resolve()
    artifacts = json.loads((root / "artifacts.json").read_text())
    indexed = {entry["path"]: entry["sha256"] for entry in artifacts["files"]}
    required = ("manifest.json", "completion.json", "operator.npz")
    for name in required:
        if name not in indexed or digest(root / name) != indexed[name]:
            raise ValueError(f"Reference integrity check failed: {name}")
    if json.loads((root / "completion.json").read_text())["status"] != "completed":
        raise ValueError("Reference preparation did not complete")
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["configuration"].get("execution_action") != "prepare":
        raise ValueError("Reference must be a completed preparation run")
    conventions = manifest["configuration"]["computation"]
    if conventions["operator"] != "pixel_line_intersections_v1" or conventions["dtype"] != "float64":
        raise ValueError("Unsupported reference operator")
    return {"run_id": manifest["run_id"], "archive": str(root),
            "files": {name: indexed[name] for name in required},
            "artifacts_sha256": digest(root / "artifacts.json"),
            "geometry": manifest["configuration"]["geometry"],
            "conventions": conventions}


def load_operator(root, reference):
    path = Path(root) / "operator.npz"
    if digest(path) != reference["files"]["operator.npz"]:
        raise ValueError("Operator changed after reference validation")
    geometry = Geometry(**reference["geometry"])
    with np.load(path, allow_pickle=False) as arrays:
        operator = SparseRadon(geometry, arrays["rows"], arrays["cols"], arrays["values"])
        if tuple(arrays["shape"]) != operator.shape:
            raise ValueError("Operator shape differs from recorded geometry")
    if not (operator.rows.shape == operator.cols.shape == operator.values.shape):
        raise ValueError("Inconsistent sparse arrays")
    if operator.values.ndim != 1 or not np.isfinite(operator.values).all():
        raise ValueError("Invalid sparse coefficients")
    if (np.any(operator.rows < 0) or np.any(operator.rows >= operator.shape[0])
            or np.any(operator.cols < 0) or np.any(operator.cols >= operator.shape[1])):
        raise ValueError("Sparse index out of bounds")
    return operator


def make_config(reference, seed, device, repetitions):
    if repetitions < 1 or seed < 0 or not device.startswith("cuda:") or not device[5:].isdigit():
        raise ValueError("Require a nonnegative seed, positive repetitions and cuda:<index>")
    return {
        "schema_version": 1, "purpose": "A20.11 CUDA matrix and input-gradient diagnostics; no training",
        "dataset": {"split": "operator_diagnostic", "split_ids": {"operator_diagnostic": ["gaussian-x-y"]},
                    "distribution": "independent standard normal entries; x then y from one PCG64 stream",
                    "reference_operator": reference},
        "geometry": reference["geometry"],
        "randomness": {"seed": seed, "generator": "NumPy PCG64; full inputs saved"},
        "computation": {"backend": "torch_cuda", "device": device, "dtype": "float64",
                        "layout": "coalesced COO; independently coalesced transpose",
                        "batch_size": 1, "warmup": 3, "repetitions": repetitions,
                        "relative_tolerance": 1e-10, "deterministic_algorithms": True,
                        "tf32": False, "cublas_workspace_config": ":4096:8",
                        "loss": "0.5 * data_norm_weight * sum((A x)^2)",
                        "gradient_convention": "Euclidean gradient = data_norm_weight * A.T @ A @ x",
                        "weighted_adjoint": "A.T * data_norm_weight / pixel_norm_weight",
                        "not_performed": ["pseudoinverse", "nullspace projection", "network training", "reconstruction"]},
    }


def benchmark(archive, config, root):
    options = config["computation"]
    # Configure before importing torch or creating a CUDA context.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = options["cublas_workspace_config"]
    import torch

    device = torch.device(options["device"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no CPU substitution is permitted")
    torch.cuda.set_device(device)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(config["randomness"]["seed"])
    properties = torch.cuda.get_device_properties(device)
    maps = Path("/proc/self/maps")
    cuda_libraries = sorted({line.split()[-1] for line in maps.read_text().splitlines()
                             if "/" in line and any(name in line for name in
                             ("libcuda", "libcusparse", "libnvJitLink", "libcublas", "libcudnn"))}) if maps.is_file() else []
    write_json(archive.path / "cuda_environment.json", {
        "torch": str(torch.__version__), "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "device": str(device),
        "name": properties.name, "total_memory_bytes": properties.total_memory,
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "visible_device_count": torch.cuda.device_count(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "torch_build": torch.__config__.show(),
        "loaded_cuda_libraries": cuda_libraries, "process_id": os.getpid(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
    })
    reference = config["dataset"]["reference_operator"]
    operator = load_operator(root, reference)
    reference_dir = archive.path / "reference"
    reference_dir.mkdir()
    for name in ("manifest.json", "completion.json", "artifacts.json", "operator.npz"):
        shutil.copyfile(Path(root) / name, reference_dir / name)
    random = np.random.Generator(np.random.PCG64(config["randomness"]["seed"]))
    x_np = random.normal(size=operator.shape[1])
    y_np = random.normal(size=operator.shape[0])
    np.savez_compressed(archive.path / "inputs.npz", x=x_np, y=y_np)
    forward_cpu = operator.forward(x_np)
    transpose_cpu = operator.transpose(y_np)
    wy = operator.geometry.data_weight
    wx = (2 / operator.geometry.size) ** 2
    gradient_cpu = wy * operator.transpose(forward_cpu)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    matrix = torch.sparse_coo_tensor(
        torch.from_numpy(np.stack((operator.rows, operator.cols)).astype(np.int64)),
        torch.from_numpy(operator.values), operator.shape, dtype=torch.float64,
    ).coalesce().to(device)
    transpose = matrix.transpose(0, 1).coalesce()
    x = torch.tensor(x_np[:, None], device=device, dtype=torch.float64)
    y = torch.tensor(y_np[:, None], device=device, dtype=torch.float64)
    torch.cuda.synchronize(device)
    conversion_seconds = time.perf_counter() - start
    archive.event("cuda_operator_ready", nnz=matrix._nnz(), geometry=asdict(operator.geometry),
                  conversion_and_transfer_seconds=conversion_seconds)

    def forward():
        return torch.sparse.mm(matrix, x)

    def transposed():
        return torch.sparse.mm(transpose, y)

    def loss_gradient():
        variable = x.detach().requires_grad_(True)
        result = torch.sparse.mm(matrix, variable)
        loss = 0.5 * wy * result.square().sum()
        return torch.autograd.grad(loss, variable)[0]

    operations = {"forward": forward, "transpose": transposed, "loss_forward_backward": loss_gradient}
    results = {name: fn().detach().cpu().numpy().ravel() for name, fn in operations.items()}
    lhs = float(np.dot(results["forward"], y_np) * wy)
    rhs = float(np.dot(x_np, results["transpose"] * wy / wx) * wx)

    def relative_error(actual, expected):
        return float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-30))

    errors = {"forward_relative_error": relative_error(results["forward"], forward_cpu),
              "transpose_relative_error": relative_error(results["transpose"], transpose_cpu),
              "loss_gradient_relative_error": relative_error(results["loss_forward_backward"], gradient_cpu),
              "weighted_adjoint_defect": abs(lhs-rhs) / max(abs(lhs), abs(rhs), 1.0)}
    np.savez_compressed(archive.path / "outputs.npz", **results,
                        cpu_forward=forward_cpu, cpu_transpose=transpose_cpu, cpu_gradient=gradient_cpu)
    write_json(archive.path / "numerical_checks.json", errors)
    archive.event("numerical_checks", **errors)
    if any(not np.isfinite(value) or value > options["relative_tolerance"] for value in errors.values()):
        raise ArithmeticError(f"CUDA numerical check failed: {errors}")
    timings = {}
    for name, fn in operations.items():
        for _ in range(options["warmup"]):
            fn()
        torch.cuda.synchronize(device)
        elapsed = []
        for _ in range(options["repetitions"]):
            start = time.perf_counter()
            fn()
            torch.cuda.synchronize(device)
            elapsed.append(time.perf_counter() - start)
        timings[name] = elapsed
        archive.event("timing", operation=name, seconds=elapsed)
    summary = {"gpu_name": properties.name, "dtype": "float64", "batch_size": 1,
               "operator_shape": operator.shape, "operator_nnz": matrix._nnz(),
               "numerical_checks": errors, "timings_seconds": timings,
               "conversion_and_transfer_seconds": conversion_seconds,
               "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
               "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
               "memory_scope": "This PyTorch process; excludes CUDA context and other processes",
               "not_performed": options["not_performed"]}
    write_json(archive.path / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-archive", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--mirror-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repetitions", type=int, default=20)
    args = parser.parse_args()
    reference = reference_metadata(args.reference_archive)
    config = make_config(reference, args.seed, args.device, args.repetitions)
    with RunArchive(config, args.runs_root, SOURCE_ROOT, args.mirror_root,
                    parent_run_id=reference["run_id"]) as archive:
        print(f"Run archive: {archive.path}", flush=True)
        benchmark(archive, config, args.reference_archive)


if __name__ == "__main__":
    main()
