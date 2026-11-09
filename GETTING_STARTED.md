# Getting started and adapting the experiments

Run commands from the repository root. Use Python 3.12. Linux is required for the original GPU controller; macOS supports the small CPU checks. Training/archive modules use POSIX facilities (`fcntl`, signals and hard links); native Windows is not a supported training environment. On Windows use a suitable Linux/WSL environment. No package installation of this repository is required: invoke its modules from the checkout.

## Choose an installation

| Goal | Requirements | Additional inputs |
| --- | --- | --- |
| Generate small diagnostic phantoms and check operators | `requirements-experiments.txt` | None for the included small checks |
| Render the paper tables and figures | `requirements-report.txt` | Included scientific export |
| Train/evaluate networks, run CPU qualifications and generate all report types | `requirements-training.txt` | Matching datasets, operators, projections and configurations; GPU runtime for GPU jobs |

The first two files are deliberately minimal. The training profile includes every directly imported third-party Python package used by the current code, including SciPy for statistical cross-checks and Pillow for image output. It is not a frozen environment with every transitive library or driver. The Pillow version reflects the current reporting environment, not a claim about every historical training run.

For paper rendering, follow the short commands in the [main README](README.md). For training and CPU checks:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
```

On Linux, choose the PyTorch build **before** installing the full profile. For CPU-only use:

```sh
.venv/bin/python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements-training.txt
```

On macOS, install `requirements-training.txt` directly; use CPU for the qualified checks, not MPS. For a Linux CUDA 12.8 environment corresponding to the study's PyTorch build, replace the CPU-wheel command above with:

```sh
.venv/bin/python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -r requirements-training.txt
```

See [PyTorch's version-specific installation instructions](https://pytorch.org/get-started/previous-versions/) for supported wheels. A suitable NVIDIA driver, accessible GPU and sufficient memory are separate prerequisites; pip does not configure these. GPU distributions and NVIDIA libraries have their own [license conditions](THIRD_PARTY_NOTICES.md).

Verify the installation:

```sh
.venv/bin/python -m pip check
.venv/bin/python -c "import torch, numpy, scipy, PIL, matplotlib; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

`False` for CUDA availability is expected for a CPU installation. For GPU runs it must be `True`; the code does not silently fall back to CPU. Native factorization additionally requires discoverable cuSOLVER, optionally selected with `DPNN_CUSOLVER_LIBRARY` as described in the [configuration guide](supplement/reproducibility.md#local-path-configuration).

## Verify a working installation

These commands generate local, ignored archives. Use a fresh output directory for independent checks.

```sh
.venv/bin/python -m dpnn_experiments check \
  --config configs/preparation/prepare_256.json --runs-root local/checks

.venv/bin/python -m tests.training_qualification \
  --config configs/tests/training_qualification.json --root local/training-check
```

The first command runs small mathematical/archive checks, not a full 256² data-generation job. The second creates its own tiny synthetic matrix/data fixture and exercises actual CPU training, model selection, interrupted runs and resume. It intentionally injects failures; the final summary must report `all_completed: true`. It is not a paper-quality U-Net/CT experiment. Do not invoke `tests/test_preparation.py` through plain unittest: the logged entry point is required.

## Original CT study versus a new experiment

The original study uses 256² single-channel images, 120 angles, 363 detectors, fixed numerical row-space factors, cached Tikhonov/TV starts and Float64 training. The [experimental setup](supplement/experiment.md) specifies all scientific settings.

`scripts/main_study.py` is the original qualified study controller: it checks for four H200 GPUs, substantial host memory, bound input archives and completed qualification records. It is not a general single-GPU launcher. Relocating paths does not make those checks disappear. For a new single-device experiment, use the underlying `dpnn_experiments.training` entry point once its input contracts have been satisfied.

The included templates contain references to external artifacts. `scripts/configure.py` resolves local paths; it does not download weights, build projection factors or convert a folder of images. Exact replay requires original source snapshots as well as data/checkpoints. See the [reproducibility guide](supplement/reproducibility.md).

## Training with your own paired data

There is currently **no ready-made image-folder or scanner-data importer**. Adapting the supplied trainer requires preparing the following data and operator contract. An image directory alone is insufficient for NSN/DP, because their layers constrain measurements through a specific forward operator.

1. Define the discrete forward operator, image geometry and measurement ordering/scaling. For this CT model, network inputs/targets are pixel values; normalized image coefficients are `h*x`, and measurement samples are multiplied by `sqrt(w)`. The stored matrix acts on normalized image coefficients. Do not apply these CT factors blindly to a different acquisition model.
2. Create fixed train/validation/test splits with unique object IDs. Prepare a target and a fixed initial reconstruction for each object. Targets are required by the supervised pixel-MSE objective; real measurements without corresponding targets require a different training objective and code adaptation.
3. Write one NPZ per object, then construct the manifest described below. The training loader consumes cached arrays; it does not reconstruct starts or add new noise during training.
4. For real NSN/DP experiments, create and verify the matching numerical row-space basis/operator archives. The `block` projection path validates hashes, geometry, successful archive completion and numerical acceptance. Start from the existing operator/factorization tools for the supported CT geometry. A new scanner/operator requires an adapter and new checks. `projection.kind=fixture` is for small technical tests, not a shortcut around scientific validation.
5. Copy `configs/study/pilot_training_template.json` into an ignored local directory and fill in its dataset reference, geometry, train/validation IDs, projection references, device, seeds and training budget. Remove all unresolved artifact placeholders by supplying real inputs. Set the method to `RES`, `NSN`, `DP` (paper DP1) or `DP_TWO` (paper DP2). Production U-Net configuration is single-channel, bilinear, fixed width, no dropout. Use image dimensions compatible with four downsampling levels (multiples of 16 are the straightforward choice), and an SSIM window no larger than the image.
6. Start with a small CPU run and check shapes, forward/transpose consistency, projection leakage, data-change budgets and solver convergence. Then run the intended device/budget. Never reuse paper hashes as if they described your new data.

### Minimum arrays read by the training loader

| NPZ key | Shape per object | Meaning |
| --- | --- | --- |
| `start` | `(1, N, N)` | Fixed initial reconstruction in pixel values |
| `target` | `(1, N, N)` | Reference image in the same units |
| `measurement` | `(m,)` | Measurement vector in the operator's normalized data coordinates |
| `delta` | scalar | Noise norm/bound in those data coordinates |
| `beta` | scalar | Nonnegative additional-data-change budget; the paper uses `beta=delta` |

Use finite numeric Float64 arrays and consistent shapes. The dataset JSON contains `geometry`, disjoint `split_ids` for `train`, `validation`, `test`, and `records` containing each object's `id`, `split`, `file` reference and `noise_realization_id`. Create file references using `dpnn_experiments.training_archive.file_reference(path)` after writing the final file; this supplies the exact hash/size/path structure expected by the loader. Bind the dataset JSON in `dataset.reference` using the same helper. `dataset.used_ids` lists the selected training/validation IDs only.

This describes the **training loader's minimum contract**, not the full study archive schema. The production `block` path also needs `operator_sha256` in the dataset and matching verified projection archives. The existing synthetic evaluator additionally requires `analytic_clean` and `raster_clean`, validates their relation to the operator/noise, and consumes frozen evaluation settings. Do not fabricate unavailable clean measurements for real scanner data; adapt the evaluator and its metrics instead. `pilot_data.py` and `tests/training_qualification.py::fixture` show the full synthetic and minimal technical examples, respectively.

### Launch, inspect and resume

After creating a valid configuration at `local/configs/my_training.json`:

```sh
.venv/bin/python -m dpnn_experiments.training \
  --config local/configs/my_training.json --root local/my-training
.venv/bin/python -m dpnn_experiments.training --audit --root local/my-training
```

The run directory printed by the trainer contains manifests, metrics, completion status, `summary.json`, selected MSE/SSIM checkpoint references and `latest_checkpoint.json`. For continuation, copy the full reference object from `latest_checkpoint.json` into `execution.resume` in a new local configuration. Preserve the original run and its source/data bindings; resume creates a new linked run. Changing source code, projection contracts or other frozen settings is not an ordinary resume. Extending a completed epoch budget uses the separate hash-bound `epoch_extension` workflow.

Keep test objects out of training/model selection. The standalone evaluator is a separate command (`python -m dpnn_experiments.evaluator --help`) and requires its own qualified, frozen configuration. The paper renderer expects the original paper export and its fixed split/seed counts; it is not a generic reporting tool for arbitrary new datasets.

## Evaluate published checkpoints

`python -m dpnn_experiments.inference` evaluates trained weights on the bare artifacts, without the hash-bound study archives that training resume and the frozen evaluator require. It needs the weight files (`.pt` state dictionaries), the row-space basis `row_space_basis.npy` and the sparse operator `operator.npz` of the block factorization, the recorded projection contract (identical for all 16 paper runs and included in `configs/study/inference_paper_template.json`), and one NPZ per object with `start`, `target`, `measurement` and `delta`, optionally `analytic_clean` and `raster_clean`. Every checkpoint carries the projection contract and is refused when it does not match the loaded basis and operator; SHA-256 values given in the specification or the contract are enforced. No model selection takes place.

1. Create a sanitized sample index from the archive `dataset.json` of a Tikhonov or TV start cache, optionally copying the sample files next to it:

```sh
.venv/bin/python -m dpnn_experiments.inference make-index \
  --dataset /path/to/start-cache/dataset.json --start-kind tikhonov \
  --output /path/to/artifacts/tikhonov/samples_index.json --copy-to /path/to/artifacts/tikhonov/samples
```

2. Place the artifacts under the IDs listed in `configs/artifacts.json` (weights `artifact-005` to `artifact-035`, basis `artifact-036`, operator `artifact-037`, Tikhonov and TV sample sets `artifact-038` and `artifact-039`) and render the template with `scripts/configure.py`, or write a specification of the same form by hand.

3. Evaluate and compare with the published per-object metrics:

```sh
.venv/bin/python -m dpnn_experiments.inference run \
  --spec local/configs/study/inference_paper_template.json --output local/inference --compare-export results
```

The output directory receives `metrics.jsonl` and `metrics.csv` with the evaluator's per-object definitions (MSE, PSNR, SSIM, data change and residual, leakage, control factor, bound checks, error decomposition), `summary.json`, and with `--compare-export` a `comparison.json` with the largest deviations from `results/data/test/individual_values.csv.gz`; the command fails when a compared value deviates by more than the tolerance (relative 1e-8, absolute 1e-12 by default). Samples are matched to models by `start_kind`, and the budget is `beta_multipliers` times each object's `delta`, as in the paper. Restrict `ids` in a sample entry for spot checks: the Float64 basis of the paper study occupies 18.8 GiB and is loaded completely, so CPU evaluation needs that much memory and a full 256² evaluation of all checkpoints is a long CPU job. The entry point is qualified on the logger fixture by

```sh
.venv/bin/python -m tests.inference_qualification \
  --config configs/tests/training_qualification.json --root local/inference-check
```

which trains each method briefly, replays the archived validation metrics from the bare weights, and checks the rejection of altered weights, contracts and samples.

## Common failures

- Missing artifact or hash mismatch: supply the matching archive, or generate and bind genuinely new data for a new study; do not disable verification.
- Source mismatch on resume: execute the source snapshot associated with that run.
- H200/four-device rejection: the original controller was selected; a new single-device experiment uses the lower-level trainer and its complete inputs.
- CUDA unavailable / cuSOLVER not found: verify driver, wheel build, device visibility and library discovery; no automatic CPU substitution is made.
- Memory exhaustion: the paper's dense Float64 factors are large. Requirements files cannot guarantee that another GPU can run this configuration.
- Resume refused by `scripts/development_pilot.py` because the environment differs: the launcher requires the same Python interpreter path and identical values of `CUDA_VISIBLE_DEVICES`, `LD_LIBRARY_PATH`, `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `CUBLAS_WORKSPACE_CONFIG` and `PYTHONHASHSEED` as at launch.

Full local run archives include real machine paths and environment information. Keep them in ignored directories and publish only intentionally prepared scientific exports.
