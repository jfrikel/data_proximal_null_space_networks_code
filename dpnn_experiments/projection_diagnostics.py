"""Fully archived SVD/gradient references and 256-square projection diagnostics."""

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from dpnn_experiments.gpu_benchmark import load_operator, reference_metadata
from dpnn_experiments.operators import Geometry, build_radon
from dpnn_experiments.phantoms import pixel_means
from dpnn_experiments.projections import FixedLandweber, MatrixOperator, inverse_polynomial_native, lsqr, null_polynomial_native
from dpnn_experiments.provenance import RunArchive, write_json

SOURCE_ROOT = Path(__file__).resolve().parents[1]


def array(value):
    return value.detach().cpu().numpy()


def error(actual, expected):
    return float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-30))


def require(condition, message):
    if not condition:
        raise ArithmeticError(message)


def setup(archive, config):
    computation = config["computation"]
    torch.manual_seed(config["randomness"]["seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(computation["device"])
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA is unavailable; no CPU fallback")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    maps = Path("/proc/self/maps")
    libraries = sorted({line.split()[-1] for line in maps.read_text().splitlines()
                        if "/" in line and any(name in line for name in
                        ("libcuda", "libcusparse", "libnvJitLink", "libcublas", "libcudnn"))}) if maps.is_file() else []
    write_json(archive.path / "torch_environment.json", {
        "torch": str(torch.__version__), "cuda_build": torch.version.cuda, "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
        "loaded_cuda_libraries": libraries, "process_id": os.getpid(), "build": torch.__config__.show()})
    return device


def solve(archive, config, operator, data, name, consistent):
    directory = archive.path / name
    directory.mkdir(parents=True)
    np.savez_compressed(directory / "rhs.npz", rhs=array(data))
    settings = config["computation"]["lsqr"]
    buffer = []
    checkpoints = set(config["computation"]["checkpoints"])
    started = time.perf_counter()

    def callback(row, solution):
        buffer.append(row)
        if len(buffer) >= 50:
            archive.event("lsqr_iterations", solve=name, rows=buffer.copy())
            buffer.clear()
        if row["iteration"] in checkpoints:
            np.savez_compressed(directory / f"iterate_{row['iteration']:05d}.npz", solution=array(solution))
        if row["iteration"] and row["iteration"] % 1000 == 0:
            print(f"{name}: iteration {row['iteration']}, residual estimate {row['estimated_residual_norm']:.3e}", flush=True)

    solution, result = lsqr(operator, data, require_data_residual=consistent, callback=callback, **settings)
    if buffer:
        archive.event("lsqr_iterations", solve=name, rows=buffer)
    if data.device.type == "cuda":
        torch.cuda.synchronize(data.device)
    result["wall_seconds_including_diagnostics"] = time.perf_counter() - started
    result["require_data_residual"] = consistent
    write_json(directory / "trace.json", result)
    np.savez_compressed(directory / "solution.npz", solution=array(solution))
    brief = {key: value for key, value in result.items() if key != "history"}
    brief["final_residuals"] = result["history"][-1]
    archive.event("lsqr_finished", solve=name, **brief)
    return solution, brief


def dp_metrics(operator, value, null, fractions):
    with torch.no_grad():
        av = operator.forward(value)
        an = operator.forward(null)
        av_norm = float(torch.linalg.vector_norm(av))
        leak = float(torch.linalg.vector_norm(an))
        records = []
        for fraction in fractions:
            beta = fraction * av_norm
            t = min(1.0, beta / av_norm) if av_norm else 1.0
            actual = float(torch.linalg.vector_norm(operator.forward(null + t * (value - null))))
            bound = beta + (1 - t) * leak
            tolerance = 1e-11 * max(1.0, av_norm, bound)
            require(actual <= bound + tolerance, "DP leakage bound failed")
            records.append({"beta_fraction": fraction, "beta": beta, "t": t,
                            "data_change": actual, "bound_with_leak": bound,
                            "excess_over_beta": max(0.0, actual - beta),
                            "leak_within_5_percent_beta": leak <= 0.05 * beta if beta else None})
        return {"input_data_norm": av_norm, "null_leak_norm": leak, "dp": records}


def gradient_checks(archive, config, device, rng):
    directory = archive.path / "gradients"
    directory.mkdir()
    # Deliberately rectangular and nonsymmetric, with a true nullspace.
    matrix = rng.normal(size=(5, 7))
    matrix[-1] = matrix[0] - 0.4 * matrix[1]
    op = MatrixOperator.dense(matrix, device)
    inverse = np.linalg.pinv(matrix, rcond=config["computation"]["svd_rtol"])
    null = np.eye(7) - inverse @ matrix
    inputs = rng.normal(size=(7, 2))
    data = rng.normal(size=(5, 2))
    np.savez_compressed(directory / "inputs.npz", matrix=matrix, inputs=inputs, data=data,
                        inverse=inverse, null=null)
    options = config["computation"]["gradcheck"]
    records = []
    for iterations in config["computation"]["gradient_iterations"]:
        layer = FixedLandweber(op, iterations)
        for name, values, custom, native in (
            ("null", inputs, layer.null, lambda z: null_polynomial_native(z, op, layer.step, iterations)),
            ("inverse", data, layer.inverse, lambda z: inverse_polynomial_native(z, op, layer.step, iterations)),
        ):
            x = torch.tensor(values, device=device, dtype=torch.float64, requires_grad=True)
            upstream = torch.tensor(rng.normal(size=tuple(custom(x).shape)), device=device, dtype=torch.float64)
            np.savez_compressed(directory / f"{name}_{iterations}_upstream.npz", upstream=array(upstream))
            custom_grad = torch.autograd.grad((custom(x) * upstream).sum(), x)[0]
            native_result = native(x)
            native_grad = (torch.autograd.grad((native_result * upstream).sum(), x)[0]
                           if native_result.requires_grad else torch.zeros_like(x))
            relative = error(array(custom_grad), array(native_grad))
            require(relative < 1e-10 or np.linalg.norm(array(custom_grad-native_grad)) < 1e-12,
                    "Custom gradient differs from unrolled finite-iteration gradient")
            passed = torch.autograd.gradcheck(custom, (x,), **options)
            record = {"method": name, "iterations": iterations, "step": layer.step,
                      "custom_vs_unrolled_relative_error": relative, "finite_difference_gradcheck": passed}
            records.append(record)
            archive.event("gradient_check", **record)
    for name, values, linear in (("svd_null", inputs, null), ("svd_inverse", data, inverse)):
        fixed = torch.tensor(linear, dtype=torch.float64, device=device)
        x = torch.tensor(values, dtype=torch.float64, device=device, requires_grad=True)
        passed = torch.autograd.gradcheck(lambda z: fixed @ z, (x,), **options)
        records.append({"method": name, "finite_difference_gradcheck": passed})
    # Undefined adaptive-solver backward must be rejected explicitly.
    try:
        lsqr(op, torch.ones(5, dtype=torch.float64, device=device, requires_grad=True),
             require_data_residual=False, **config["computation"]["lsqr"])
    except ValueError as exc:
        require("autograd" in str(exc), "Unexpected adaptive-gradient guard error")
    else:
        raise ArithmeticError("Adaptive LSQR silently accepted a gradient input")
    write_json(directory / "checks.json", records)
    return records


def boundary_checks(archive, config, device):
    settings = config["computation"]
    records = []
    for name, matrix in (("zero_operator", np.zeros((3, 4))),
                         ("full_column_rank", np.array([[1., 0.], [0., 2.], [1., 1.]]))):
        op = MatrixOperator.dense(matrix, device)
        rhs = np.array([1., -2., 0.5])
        solution, result = lsqr(op, torch.tensor(rhs, device=device), require_data_residual=False, **settings["lsqr"])
        expected = np.linalg.pinv(matrix) @ rhs
        np.savez_compressed(archive.path/f"boundary_{name}.npz", matrix=matrix,rhs=rhs,solution=array(solution),expected=expected)
        absolute = float(np.linalg.norm(array(solution)-expected))
        require(absolute < 1e-10, "Boundary LSQR reference failed")
        if name == "zero_operator":
            v = torch.arange(4,dtype=torch.float64,device=device)
            layer = FixedLandweber(op,8)
            require(torch.equal(layer.null(v),v), "Zero operator null polynomial is not identity")
            require(bool(torch.all(layer.inverse(torch.tensor(rhs,device=device))==0)), "Zero operator inverse is not zero")
        records.append({"case":name,"absolute_inverse_error":absolute,"solver":result})
    write_json(archive.path/"boundary_checks.json",records)
    return records


def reference(archive, config, device):
    rng = np.random.Generator(np.random.PCG64(config["randomness"]["seed"]))
    settings = config["computation"]
    cases = []
    left, _ = np.linalg.qr(rng.normal(size=(7, 4)))
    right, _ = np.linalg.qr(rng.normal(size=(9, 4)))
    synthetic = (left * np.array(config["dataset"]["synthetic_singular_values"])) @ right.T
    cases.append(("controlled_rank_deficient", synthetic, None))
    for geometry_config in config["geometry"]["references"]:
        geometry = Geometry(**geometry_config)
        radon = build_radon(geometry)
        radon.save(archive.path / f"radon_{geometry.size}.npz")
        matrix = radon.dense() * math.sqrt(geometry.data_weight) / (2 / geometry.size)
        cases.append((f"radon_{geometry.size}", matrix, asdict(geometry)))
    summaries = []
    for name, matrix, geometry in cases:
        directory = archive.path / name
        directory.mkdir()
        archive.event("svd_started", case=name, shape=matrix.shape, geometry=geometry)
        u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
        cutoff = settings["svd_rtol"] * singular[0]
        keep = singular > cutoff
        inverse = (vh[keep].T / singular[keep]) @ u[:, keep].T
        null = np.eye(matrix.shape[1]) - vh[keep].T @ vh[keep]
        q = np.eye(matrix.shape[1]) - null
        identities = {"ABA_relative": error(matrix @ inverse @ matrix, matrix),
                      "BAB_relative": error(inverse @ matrix @ inverse, inverse),
                      "AB_symmetry_relative": error(matrix @ inverse, (matrix @ inverse).T),
                      "BA_symmetry_relative": error(inverse @ matrix, (inverse @ matrix).T),
                      "P_idempotence_relative": error(null @ null, null),
                      "P_symmetry_relative": error(null, null.T),
                      "AP_norm": float(np.linalg.norm(matrix @ null, 2)),
                      "P_plus_Q_identity_norm": float(np.linalg.norm(null+q-np.eye(len(null))))}
        require(max(identities.values()) < settings["identity_tolerance"], "SVD reference identity check failed")
        random = rng.normal(size=matrix.shape[1]); random /= np.linalg.norm(random)
        kernel = null @ rng.normal(size=matrix.shape[1]); kernel /= np.linalg.norm(kernel)
        probes = np.column_stack((random, vh[0], vh[np.flatnonzero(keep)[-1]], kernel))
        rhs = rng.normal(size=matrix.shape[0]); rhs /= np.linalg.norm(rhs)
        np.savez_compressed(directory / "reference.npz", matrix=matrix, u=u, singular=singular, vh=vh,
                            inverse=inverse, null=null, probes=probes, arbitrary_rhs=rhs)
        op = MatrixOperator.dense(matrix, device)
        values = torch.tensor(probes, dtype=torch.float64, device=device)
        filters = []
        for iterations in settings["landweber_iterations"]:
            layer = FixedLandweber(op, iterations)
            actual = array(layer.null(values))
            suppression = -np.expm1(iterations * np.log1p(-layer.step * singular**2))
            expected = probes - vh.T @ (suppression[:, None] * (vh @ probes))
            filter_error = error(actual, expected)
            require(filter_error < settings["filter_tolerance"], "Landweber differs from its SVD polynomial")
            inverse_actual = array(layer.inverse(torch.tensor(rhs, device=device, dtype=torch.float64)))
            inverse_filter = np.divide(suppression, singular, out=np.zeros_like(singular), where=singular > 0)
            inverse_expected = vh.T @ (inverse_filter * (u.T @ rhs))
            inverse_filter_error = error(inverse_actual, inverse_expected)
            require(inverse_filter_error < settings["filter_tolerance"] or iterations == 0,
                    "Landweber inverse differs from its SVD polynomial")
            record = {"iterations": iterations, "step": layer.step, "spectral_filter_relative_error": filter_error,
                      "inverse_spectral_filter_relative_error": inverse_filter_error,
                      "null_error_by_probe": [float(np.linalg.norm(actual[:,j]-null@probes[:,j])) for j in range(4)],
                      "probe_order": ["random", "largest_singular", "smallest_retained_singular", "numerical_kernel"]}
            filters.append(record)
            np.savez_compressed(directory / f"landweber_{iterations}.npz", null=actual, inverse=inverse_actual)
        solver_checks = []
        for label, b, consistent in (("matched", matrix @ random, True), ("arbitrary", rhs, False)):
            solution, trace = solve(archive, config, op, torch.tensor(b, device=device, dtype=torch.float64),
                                    f"{name}/lsqr_{label}", consistent)
            actual = array(solution)
            relative = error(actual, inverse @ b)
            record = {"rhs": label, "image_error_against_svd": relative,
                      "kernel_component_norm": float(np.linalg.norm(null @ actual)),
                      "meets_image_target": relative <= settings["image_target"], **trace}
            write_json(directory/f"lsqr_{label}"/"svd_comparison.json",record)
            archive.event("lsqr_svd_comparison",case=name,**record)
            require(relative < settings["image_target"], f"LSQR fails the {name}/{label} SVD image target")
            solver_checks.append(record)
        for label, b in (("zero_rhs", np.zeros(matrix.shape[0])),):
            solution, _ = lsqr(op, torch.tensor(b, device=device), require_data_residual=True, **settings["lsqr"])
            require(bool(torch.all(solution == 0)), "Zero RHS did not return zero")
        dp = dp_metrics(op, values[:, 0], torch.tensor(null @ random, device=device), settings["beta_fractions"])
        record = {"case": name, "shape": matrix.shape, "numerical_rank": int(keep.sum()),
                  "cutoff": float(cutoff), "smallest_retained_singular": float(singular[keep][-1]),
                  "norm_bound_squared": op.norm_bound_squared, "identities": identities,
                  "landweber": filters, "lsqr": solver_checks, "svd_dp": dp}
        summaries.append(record)
        write_json(directory / "summary.json", record)
        print(f"{name}: rank {keep.sum()}, LSQR image errors {[r['image_error_against_svd'] for r in solver_checks]}", flush=True)
    # Cutting a positive singular value yields a projector for a modified matrix.
    u,s,vh=np.linalg.svd(synthetic,full_matrices=False)
    cutoff=settings["positive_cutoff_fraction"]*s[0]
    basis=vh[s>cutoff]
    trunc_null=np.eye(synthetic.shape[1])-basis.T@basis
    leakage=float(np.linalg.norm(synthetic@trunc_null,2))
    require(0 < leakage <= cutoff+1e-12,"Positive-singular-value cutoff check failed")
    np.savez_compressed(archive.path/"positive_cutoff.npz",matrix=synthetic,projector=trunc_null,singular=s)
    truncation={"cutoff":float(cutoff),"original_operator_leak_norm":leakage,
                "qualification":"Idempotent numerical projector, but not a kernel projector for the original matrix"}
    gradients = gradient_checks(archive, config, device, rng)
    boundaries = boundary_checks(archive,config,device)
    summary = {"mode": "reference", "references": summaries, "gradients": gradients,
               "boundary_checks":boundaries,"positive_cutoff":truncation,
               "limitations": "Numerical SVD rank uses an explicit cutoff; finite Landweber is a filter, not an exact projector."}
    write_json(archive.path / "summary.json", summary)


def directional_check(archive, config, function, value, upstream, direction, name):
    eps = config["computation"]["directional_epsilon"]
    x = value.detach().requires_grad_(True)
    start = time.perf_counter()
    result = function(x)
    grad = torch.autograd.grad((result * upstream).sum(), x)[0]
    with torch.no_grad():
        finite = float(((function(x+eps*direction)-function(x-eps*direction))*upstream).sum() / (2*eps))
        analytic = float((grad*direction).sum())
    if x.device.type == "cuda":
        torch.cuda.synchronize(x.device)
    absolute = abs(analytic-finite)
    relative = absolute/max(abs(analytic), abs(finite), 1e-12)
    record = {"name": name, "epsilon": eps, "finite_difference": finite, "autograd": analytic,
              "absolute_error": absolute, "relative_error": relative,
              "wall_seconds_including_three_forwards_and_backward": time.perf_counter()-start}
    np.savez_compressed(archive.path/f"gradient_{name}.npz", input=array(x), output=array(result),
                        upstream=array(upstream), direction=array(direction), gradient=array(grad))
    archive.event("directional_gradient", **record)
    require(absolute < 1e-9 + 1e-5 * max(abs(analytic),abs(finite)), "Large gradient finite-difference check failed")
    return record


def large(archive, config, device, reference_root):
    ref = config["dataset"]["reference_operator"]
    radon = load_operator(reference_root, ref)
    directory = archive.path / "reference"
    directory.mkdir()
    for name in ("manifest.json", "completion.json", "artifacts.json", "operator.npz"):
        shutil.copyfile(reference_root/name, directory/name)
    op = MatrixOperator.whitened_radon(radon, device)
    settings = config["computation"]
    rng = np.random.Generator(np.random.PCG64(config["randomness"]["seed"]))
    random = rng.normal(size=op.shape[1]); random /= np.linalg.norm(random)
    circle = pixel_means({"ellipses": [config["dataset"]["circle"]]}, radon.geometry.size,
                         config["dataset"]["pixel_quadrature"]).ravel()
    circle /= np.linalg.norm(circle)
    probes = np.column_stack((random, circle))
    values = torch.tensor(probes, device=device, dtype=torch.float64)
    np.savez_compressed(archive.path/"inputs.npz", probes=probes)
    step = FixedLandweber(op, 1).step
    write_json(archive.path/"operator_contract.json", {
        "shape": op.shape, "nnz": op.matrix._nnz(), "norm_bound_squared": op.norm_bound_squared,
        "norm_bound_method": "max absolute row sum * max absolute column sum of whitened M",
        "landweber_step": step, "pixel_weight": (2/radon.geometry.size)**2,
        "data_weight": radon.geometry.data_weight,
        "coordinate_contract": "M=sqrt(wY/wX)*A; all solver inputs/outputs are whitened; each test image has unit whitened norm"})
    filters = []
    with torch.no_grad():
        state = values.clone()
        completed = 0
        started = time.perf_counter()
        for target in settings["landweber_iterations"]:
            state = null_polynomial_native(state, op, step, target-completed)
            completed = target
            np.savez_compressed(archive.path/f"landweber_{target}.npz", null=array(state))
            twice = FixedLandweber(op, target, step).null(state)
            for j, name in enumerate(config["dataset"]["probe_names"]):
                record = {"probe": name, "iterations": target,
                          "idempotence_defect_norm": float(torch.linalg.vector_norm(twice[:,j]-state[:,j])),
                          "null_component_norm": float(torch.linalg.vector_norm(state[:,j])),
                          **dp_metrics(op, values[:,j], state[:,j], settings["beta_fractions"])}
                filters.append(record)
                archive.event("landweber_diagnostic", **record)
            print(f"Landweber: {target} iterations completed", flush=True)
        landweber_seconds = time.perf_counter()-started
    solvers = []
    for j, name in enumerate(config["dataset"]["probe_names"]):
        value = values[:,j]
        q, trace = solve(archive, config, op, op.forward(value), f"lsqr_{name}", True)
        null = value-q
        reprojection, second_trace = solve(archive, config, op, op.forward(null), f"lsqr_{name}_reprojection", True)
        record = {"probe": name, "solver": trace, "reprojection_solver": second_trace,
                  "idempotence_defect_norm": float(torch.linalg.vector_norm(reprojection)),
                  "orthogonality_inner_product": float(torch.dot(null,q)),
                  "null_component_norm": float(torch.linalg.vector_norm(null)),
                  "range_component_norm": float(torch.linalg.vector_norm(q)),
                  **dp_metrics(op,value,null,settings["beta_fractions"])}
        solvers.append(record)
        np.savez_compressed(archive.path/f"decomposition_{name}.npz", q=array(q), null=array(null),
                            reprojected_range=array(reprojection))
        archive.event("lsqr_projection_diagnostic", **record)
    gradients = []
    direction = torch.tensor(rng.normal(size=op.shape[1]),device=device,dtype=torch.float64)
    direction /= torch.linalg.vector_norm(direction)
    upstream = torch.tensor(rng.normal(size=op.shape[1]),device=device,dtype=torch.float64)
    upstream /= torch.linalg.vector_norm(upstream)
    for count in settings["gradient_iterations"]:
        layer = FixedLandweber(op,count,step)
        gradients.append(directional_check(archive,config,layer.null,values[:,0],upstream,direction,f"null_{count}"))
    data = torch.tensor(rng.normal(size=op.shape[0]),device=device,dtype=torch.float64)
    data /= torch.linalg.vector_norm(data)
    data_direction = torch.tensor(rng.normal(size=op.shape[0]),device=device,dtype=torch.float64)
    data_direction /= torch.linalg.vector_norm(data_direction)
    inverse_layer = FixedLandweber(op,settings["inverse_gradient_iterations"],step)
    gradients.append(directional_check(archive,config,inverse_layer.inverse,data,upstream,data_direction,"inverse"))
    summary = {"mode": "large", "shape": op.shape, "norm_bound_squared": op.norm_bound_squared,
               "step": step, "landweber": filters, "landweber_wall_seconds_including_all_diagnostics": landweber_seconds,
               "lsqr": solvers, "gradients": gradients,
               "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None,
               "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type=="cuda" else None,
               "limitations": ["No large SVD or certified large image-space projection error",
                               "Adaptive LSQR has no gradient contract", "Finite Landweber is not an exact nullspace projector",
                               "Two development inputs; no network or training"]}
    write_json(archive.path/"summary.json",summary)
    print(json.dumps({"lsqr":solvers,"gradients":gradients,"peak_allocated_bytes":summary["peak_allocated_bytes"]},indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--reference-archive",type=Path)
    parser.add_argument("--runs-root",type=Path,required=True)
    parser.add_argument("--mirror-root",type=Path,required=True)
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    options=config["computation"]
    if options["dtype"]!="float64" or config["mode"] not in ("reference","large"):
        raise ValueError("Unsupported mode or dtype")
    checkpoints=options["landweber_iterations"]
    if checkpoints!=sorted(set(checkpoints)) or any(type(k) is not int or k<0 for k in checkpoints):
        raise ValueError("Landweber checkpoints must be sorted unique nonnegative integers")
    if config["mode"]=="large":
        if args.reference_archive is None:
            raise ValueError("Large diagnostics require an archived reference operator")
        ref=reference_metadata(args.reference_archive)
        if ref["geometry"]!=config["geometry"]:
            raise ValueError("Configured geometry differs from reference")
        config["dataset"]["reference_operator"]=ref
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=options["cublas_workspace_config"]
    with RunArchive(config,args.runs_root,SOURCE_ROOT,args.mirror_root,
                    parent_run_id=config["dataset"].get("reference_operator",{}).get("run_id")) as archive:
        print(f"Run archive: {archive.path}",flush=True)
        device=setup(archive,config)
        if config["mode"]=="reference":
            reference(archive,config,device)
        else:
            large(archive,config,device,args.reference_archive)


if __name__=="__main__":
    main()
