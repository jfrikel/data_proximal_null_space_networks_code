# Implementation map

## Scientific workflow

| Stage | Main module / entry point |
|---|---|
| Local path configuration | `scripts/configure.py` |
| Phantom generation and discretization | `dpnn_experiments/phantoms.py`, `operators.py`, `pilot_data.py` |
| Tikhonov/TV initializations | `start_preparation.py`, `reconstruction.py`, `cached_starts.py` |
| Numerical row-space basis | `tools/factorization_diagnostics.py`, `dpnn_experiments/block_factorization.py`, `factor_verification.py` |
| Networks and training | `dpnn_experiments/network_layers.py`, `training.py`, `src/unet.py` |
| Study orchestration | `scripts/main_study.py`, `dpnn_experiments/study_plan.py`, `study_queue.py` |
| Historical continuation | `dpnn_experiments/epoch_extension.py` |
| Evaluation | `dpnn_experiments/evaluator.py`, `paired_statistics.py` |
| Paper reproduction | `scripts/reproduce_paper.py` |

Paths in the workflow table are relative to the repository root unless a filename follows a module in the same directory. The main controller dynamically loads `scripts/development_pilot.py`. Operator-loading helpers in `gpu_benchmark.py`, the Tikhonov solver in `pilot_data.py`, and numerical helpers in `projection_diagnostics.py` are active dependencies despite their historical names.

`configs/study/` holds execution templates; `configs/preparation/` holds operator/factorization settings; `configs/tests/` and `configs/diagnostics/` hold technical check settings. `configs/paper/` records completed-study settings and is descriptive rather than directly executable. See the [path configuration guide](reproducibility.md#local-path-configuration).

## Python modules

Numerical tests are separate from the scientific implementation. Some qualification modules require GPU artifacts; module presence does not imply that every check can run on a CPU-only installation. The source snapshot includes the relevant test and tool directories so that integrity checks remain explicit after reorganization.

### dpnn_experiments

| File | Purpose |
|---|---|
| [dpnn_experiments/__init__.py](../dpnn_experiments/__init__.py) | Independent experiment preparation; legacy training entry points are unchanged. |
| [dpnn_experiments/__main__.py](../dpnn_experiments/__main__.py) | Run with python -m dpnn_experiments {check,prepare,verify}. |
| [dpnn_experiments/acceleration.py](../dpnn_experiments/acceleration.py) | Fixed Chebyshev filters and left row scaling in whitened coordinates. |
| [dpnn_experiments/block_factorization.py](../dpnn_experiments/block_factorization.py) | Two-block row-space factorization, avoiding a single oversized SVD. |
| [dpnn_experiments/cached_starts.py](../dpnn_experiments/cached_starts.py) | Strict consumption of hash-bound start_preparation datasets; no solver fallback. |
| [dpnn_experiments/cuda_svd.py](../dpnn_experiments/cuda_svd.py) | Minimal Float64 binding to the documented cuSOLVER 64-bit Xgesvd API. |
| [dpnn_experiments/development_pilot.py](../dpnn_experiments/development_pilot.py) | Sequential development study; stage processes isolate CUDA memory and archives. |
| [dpnn_experiments/epoch_extension.py](../dpnn_experiments/epoch_extension.py) | Explicit, hash-bound continuation of a completed epoch budget. |
| [dpnn_experiments/evaluator.py](../dpnn_experiments/evaluator.py) | Standalone frozen validation/test evaluator with certified start-solver stops. |
| [dpnn_experiments/factor_verification.py](../dpnn_experiments/factor_verification.py) | Full-matrix residual checks and application benchmark of archived factors. |
| [dpnn_experiments/gpu_benchmark.py](../dpnn_experiments/gpu_benchmark.py) | Logged CUDA checks of an archived Radon matrix, without training or inversion. |
| [dpnn_experiments/main_inputs.py](../dpnn_experiments/main_inputs.py) | Single-writer input preparation recovery without changing solver settings. |
| [dpnn_experiments/main_integrity.py](../dpnn_experiments/main_integrity.py) | Fail-closed bindings for main-study approvals and validation selections. |
| [dpnn_experiments/network_layers.py](../dpnn_experiments/network_layers.py) | Float64 NSN/scalar-DP layers using fixed, provenance-bound row-space factors. |
| [dpnn_experiments/operators.py](../dpnn_experiments/operators.py) | Pixelwise-constant line integrals with an exact algebraic transpose. |
| [dpnn_experiments/paired_statistics.py](../dpnn_experiments/paired_statistics.py) | Object-paired descriptive statistics and BCa intervals for a mean effect. |
| [dpnn_experiments/phantoms.py](../dpnn_experiments/phantoms.py) | Continuous ellipse objects, pixel area means and independent analytic sinograms. |
| [dpnn_experiments/pilot_data.py](../dpnn_experiments/pilot_data.py) | Independent analytic ellipse data and checked Tikhonov starts for a pilot. |
| [dpnn_experiments/pilot_review.py](../dpnn_experiments/pilot_review.py) | Review existing pilot archives without inference, selection, or test-image access. |
| [dpnn_experiments/projection_acceptance.py](../dpnn_experiments/projection_acceptance.py) | Reusable Float64 projection acceptance; exact reference only on small fixtures. |
| [dpnn_experiments/projection_diagnostics.py](../dpnn_experiments/projection_diagnostics.py) | Fully archived SVD/gradient references and 256-square projection diagnostics. |
| [dpnn_experiments/projections.py](../dpnn_experiments/projections.py) | Fixed-operator inversion in Euclidean (whitened) coordinates. |
| [dpnn_experiments/provenance.py](../dpnn_experiments/provenance.py) | Durable, self-contained archives for the new preparation and diagnostic runs. |
| [dpnn_experiments/reconstruction.py](../dpnn_experiments/reconstruction.py) | Regularized inverse and isotropic TV in whitened image/data coordinates. |
| [dpnn_experiments/start_preparation.py](../dpnn_experiments/start_preparation.py) | Immutable start datasets for training: regularized inverse or isotropic TV. |
| [dpnn_experiments/start_recovery.py](../dpnn_experiments/start_recovery.py) | Reuse only qualified objects from hash-bound, closed preparation archives. |
| [dpnn_experiments/study_plan.py](../dpnn_experiments/study_plan.py) | Versioned single-noise study preparation. No numerical work at prepare time. |
| [dpnn_experiments/study_queue.py](../dpnn_experiments/study_queue.py) | Exclusive device slots and a durable DAG queue. Never chains study phases. |
| [dpnn_experiments/study_report.py](../dpnn_experiments/study_report.py) | CPU-only complete-case accounting, paired analysis and portable SVG/HTML output. |
| [dpnn_experiments/training.py](../dpnn_experiments/training.py) | Single-device Float64 training/evaluation with complete artifacts and resume. |
| [dpnn_experiments/training_archive.py](../dpnn_experiments/training_archive.py) | Durable training segments, immutable manifests and linked atomic checkpoints. |

### src

| File | Purpose |
|---|---|
| [src/__init__.py](../src/__init__.py) | Package initialization. |
| [src/unet.py](../src/unet.py) | Defines `DoubleConv`, `Down`, `Up`, `OutConv`, `UNet`. |

### scripts

| File | Purpose |
|---|---|
| [scripts/configure.py](../scripts/configure.py) | Create local configurations with explicit artifact paths; never start a run. |
| [scripts/development_pilot.py](../scripts/development_pilot.py) | Detached, immutable-source development-pilot launcher (Linux/macOS). |
| [scripts/main_study.py](../scripts/main_study.py) | Explicit preparation, validation, freeze, test, report; no phase autorun. |
| [scripts/reproduce_paper.py](../scripts/reproduce_paper.py) | Reproduce paper tables/figures from the public scientific export (no inference). |

### tests

| File | Purpose |
|---|---|
| [tests/__init__.py](../tests/__init__.py) | Numerical qualification tests. |
| [tests/cache_qualification.py](../tests/cache_qualification.py) | Manifested real small-CT cache/evaluator and periodic-retention integration tests. |
| [tests/continuation_qualification.py](../tests/continuation_qualification.py) | Archived external-cache continuation checks, including real checkpoint resume. |
| [tests/continuation_rehearsal.py](../tests/continuation_rehearsal.py) | Small archived H200 test of TV continuation with actual U-Nets and checkpoint resume. |
| [tests/development_qualification.py](../tests/development_qualification.py) | Archived CPU qualification of detached development orchestration and inherited best weights. |
| [tests/epoch_extension_qualification.py](../tests/epoch_extension_qualification.py) | CPU-only epoch-extension, checkpoint selection, and four-worker replay tests. |
| [tests/evaluator_qualification.py](../tests/evaluator_qualification.py) | Archived solver, projection and standalone evaluator qualification. |
| [tests/h200_rehearsal.py](../tests/h200_rehearsal.py) | End-to-end fixed-TV, independent-U-Net, periodic-checkpoint rehearsal. |
| [tests/retention_qualification.py](../tests/retention_qualification.py) | Periodic storage reduction and off-interval recovery, with complete archives. |
| [tests/start_qualification.py](../tests/start_qualification.py) | Small Radon end-to-end qualification of immutable TV/Tikhonov start caches. |
| [tests/start_recovery_qualification.py](../tests/start_recovery_qualification.py) | Archived recovery, exact replay and corruption rejection on small Radon data. |
| [tests/study_qualification.py](../tests/study_qualification.py) | Small CPU fixtures for study v2. Never reads scientific main test objects. |
| [tests/test_preparation.py](../tests/test_preparation.py) | Scientific and archive checks; invoke through the logged 'check' command. |
| [tests/training_qualification.py](../tests/training_qualification.py) | Archived logger, crash-window, checkpoint/resume and solver qualification. |
| [tests/two_branch_h200.py](../tests/two_branch_h200.py) | Fixed-basis two-U-Net acceptance: proximity, independent VJP and directions. |
| [tests/two_branch_qualification.py](../tests/two_branch_qualification.py) | Archived independent two-branch projection/gradient and actual U-Net checks. |

### tools

| File | Purpose |
|---|---|
| [tools/__init__.py](../tools/__init__.py) | Preparation and diagnostic command-line tools. |
| [tools/acceleration_diagnostics.py](../tools/acceleration_diagnostics.py) | Worker functions for logged fixed-filter and row-preconditioned comparisons. |
| [tools/factorization_diagnostics.py](../tools/factorization_diagnostics.py) | Supervised, fully archived factorization diagnostics for a fixed Radon matrix. |
| [tools/layer_diagnostics.py](../tools/layer_diagnostics.py) | Archived fixed-factor NSN/scalar-DP integration and derivative diagnostics. |
