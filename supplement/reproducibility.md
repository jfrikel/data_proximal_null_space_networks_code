# Reproducibility and local configuration

For installation, CPU training checks and adapting your own data, start with [Getting started](../GETTING_STARTED.md).

## Rebuild the paper tables and figures

Follow the report installation commands in the [README](../README.md), then run:

```sh
.report-venv/bin/python scripts/reproduce_paper.py
```

The renderer reads `results/data/` and writes `results/generated/`. It checks split sizes, completed training positions, scientific settings, paired object identities, selected image metrics and cost-input hashes. It produces the main paper outputs and additional seed/checkpoint comparisons. The standard deviation is the sample SD across 500 object-wise seed-averaged metrics (`ddof=1`).

Options:

- `--tables-only`: aggregate tables without rendering figures.
- `--output-dir PATH`: choose a separate output directory.
- `--data-root PATH`: point to an export root containing `data/`.
- `--cost-file PATH`: specify a relocated `cost.json` explicitly.

`results/data/export.json` describes the public derivation and records hashes of its source exports. `results/manifest.json` gives hashes and sizes of the current public files. The `source_file_sha256` values in `export.json` refer to the source exports before local paths were removed; four files (`data/examples/index.json`, `data/metadata/profile.json`, `data/metadata/training_runs.json.xz` and `data/cost.json`) therefore differ from their published versions in those hashes only. Use `results/manifest.json` to verify the checkout. Operational paths, hardware IDs and complete process environments are excluded. Original arrays and per-object metric observations are retained; training metadata is reduced to the scientific settings and summaries needed here. This is not a complete training/resume archive.

## Local path configuration

No server account or workstation path is prescribed by the repository. External inputs are represented by neutral IDs in [configs/artifacts.json](../configs/artifacts.json). Its `used_by` entries identify the configuration fields that need each artifact. Depending on the field, an ID denotes an archive directory or a specific file; recorded SHA256 values still identify the required original contents.

Choose your artifact directory and generate local copies of the execution configurations:

```sh
python3 scripts/configure.py --artifact-root /path/to/your/artifacts
```

Replace the example path with your own directory. This command writes to `local/configs/`, which is ignored by Git. It does not start an experiment, download files or alter the versioned templates. It expands `${DPNN_ARTIFACT_ROOT}`, resolves references between configuration files, and reports missing artifact paths. It refuses to overwrite existing local configurations. Use `--output-dir PATH` for another new destination.

Place the original artifacts under the indicated IDs, or explicitly edit the paths in your generated local configurations. For example, `operator_root` must point to the required operator archive; a `reference.path` must point to the referenced file, not only its parent directory. Paths and scientific hashes serve different purposes: changing a location must not change or bypass the expected content hash.

To require all artifact paths in the generated execution templates to exist, add `--check-artifacts`. Without that flag, configurations can be prepared before the large artifacts are available. The path check does not replace the full integrity checks performed by the experiment code. `configs/paper/` is descriptive metadata and is not rendered into an executable study profile.

The study controller accepts explicit locations, for example:

```sh
python3 scripts/main_study.py prepare /path/to/new/study \
  --profile local/configs/study/main_study_256.json
```

This prepares the historical 30-epoch starting profile and requires its referenced artifacts/configuration to be available. It is not a complete fresh-training recipe or a claim that the Git checkout alone can reproduce the GPU study. The study root must be outside the checkout; later stages retain their input-binding and qualification requirements. `python3 scripts/main_study.py --help` lists the explicit commands `prepare`, `bind-inputs`, `prepare-inputs`, `start`, `resume`, `status`, `stop`, `freeze-tests`, `report` and `review-rehearsal`; the validation or test phase is selected with `--phase`. Preparation does not start training.

For native cuSOLVER factorization and workspace queries, the library is located through the system library resolver. If necessary, set `DPNN_CUSOLVER_LIBRARY` to your installed cuSOLVER library file before invoking the factorization tool; this setting takes precedence over system discovery. No machine-specific library directory is embedded in the source.

For preparation checks, `--runs-root` selects the output location. Metadata mirroring is disabled by default; use `--mirror-root PATH` only if a second local copy is wanted. Full run logs can contain your actual runtime paths and machine information. Keep them in ignored local directories; do not publish a raw run archive as a substitute for a scientific export.

## Re-run inference or training

Required inputs include selected weights and their checkpoint index, the discrete operator and numerical row-space basis, test/training data as appropriate, and qualified Tikhonov/TV reconstruction caches. Selected image arrays in the public export are not all test predictions. These large artifacts are not included and no public download index is supplied.

The runtime is in `dpnn_experiments/`; `scripts/main_study.py` is the study controller. The underlying launcher `scripts/development_pilot.py` is also used by the main controller despite its historical name. Preparation diagnostics live in `tools/`, and numerical tests in `tests/`. The former `experiments_v2` configuration directory has been replaced by the grouped `configs/` tree.

The original study ran for 30 epochs and continued to 60 with optimizer, scheduler and random states preserved. No run stopped early. Epoch-30 test results were inspected before the extension; final selection used validation data but evaluation reused the same test objects. A fresh run with 60 epochs is not a byte-identical replay of this history.

The public Git history starts with the cleaned release snapshot. Commit hashes recorded in the export and in the paper metadata (`source_commit` in `results/data/export.json`, `code_commit` in `configs/paper/verified-settings.json`) refer to the private development history that preceded this snapshot and do not resolve in the public repository; they are retained because the archived study files are bound to them. Historical experiment archives bind checkpoints to their exact original source snapshot, independently of this new Git history. The reorganization changes source paths and hashes; resuming such an archive requires its original snapshot. Do not edit archived hashes or disable verification to make an old checkpoint accept a new source tree. The public entry point `python -m dpnn_experiments.inference` evaluates published weights on the bare artifacts (basis, operator, sample files) without those archives; the [getting-started guide](../GETTING_STARTED.md#evaluate-published-checkpoints) lists the required inputs, and `configs/study/inference_paper_template.json` records the paper checkpoints, their hashes and the projection contract.

The recorded compute environment used NumPy 1.26.4, PyTorch 2.7.0, SciPy 1.11.4 and a CUDA 12.8 PyTorch build. Computation used Float64 on one H200 per run, deterministic settings and disabled TF32; four GPUs supported concurrent jobs. The controller enforces the qualified four-H200 resource setup. These are documented experimental constraints, not private hostnames. `requirements-experiments.txt` covers NumPy preparation only, and `requirements-report.txt` covers report generation; neither installs the training dependencies. `requirements-training.txt` adds PyTorch, SciPy and Pillow; select the appropriate PyTorch wheel as explained in the getting-started guide. NVIDIA drivers, native runtime discovery and bound study artifacts remain separate requirements.

## Small numerical checks

With the preparation dependencies installed, run from the repository root:

```sh
python3 -m dpnn_experiments check \
  --config configs/preparation/prepare_256.json \
  --runs-root local/checks
```

This exercises small phantoms, line intersections, weighted adjoints, projection bounds and archive integrity, with logged outcomes. PyTorch-dependent qualification modules under `tests/` require an appropriate PyTorch environment; some also require SciPy or actual GPU artifacts. These checks are distinct from a new full study or a new evaluation of trained networks.

The reorganization was checked with CPU module imports, the small preparation suite, tiny-model training/resume qualification and full paper rendering. GPU training and historical checkpoint replay were not rerun.
