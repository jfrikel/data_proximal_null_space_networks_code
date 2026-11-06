# Data-proximal null-space networks for inverse problems

Research code accompanying **Data-proximal null-space networks for inverse problems** by **Simon Göppel, Jürgen Frikel, and Markus Haltmeier**. [Paper sources](https://github.com/jfrikel/data_proximal_null_space_networks).

The paper introduces data-proximal networks as an extension of null-space networks. Learned corrections explicitly control the additional change in predicted data relative to an initial reconstruction. The theoretical analysis establishes regularization and data-proximity guarantees under the stated assumptions; numerical experiments illustrate reconstruction quality and the implemented data-change control.

This repository provides the implementation and experimental documentation for the paper's synthetic limited-angle computed tomography study. Its scientific purpose is to make the numerical methods inspectable and support reproduction of the reported experiments and results. It includes synthetic ellipse generation, the discrete Radon operator and numerical projection layers, Tikhonov and TV initializations, network training and checkpoint evaluation, and scripts that regenerate tables and figures from the included scientific export.

The study compares an unconstrained residual network (RES), a null-space network (NSN), and two data-proximal variants (DP1 and DP2), using both initializations. It uses 2,500 training, 500 validation and 500 test objects, two training seeds, and 60 epochs per model. Evaluations cover MSE, PSNR, SSIM, additional data change, reconstruction residuals and computational cost. The artifact requirements for rendering results, running inference and training are distinguished below.

Both DP variants substantially improve mean reconstruction quality over NSN for both initializations. Tikhonov-initialized DP1 approaches RES quality. DP2 offers no quality advantage over DP1 in this experiment and requires approximately twice the optimization time and trainable parameters.

## Start here

- [Getting started and training with your own data](GETTING_STARTED.md): installation profiles, working CPU checks and data/operator contracts.
- [Experimental setup](supplement/experiment.md): data, discretization, networks and training.
- [Results](supplement/results.md): quality, data proximity and computational cost.
- [Reproducibility and local path configuration](supplement/reproducibility.md): commands, artifact requirements and checks.
- [Implementation map](supplement/code-map.md): entry points and Python modules.

## Reproduce the paper results

Run from the repository root with Python 3.12:

```sh
python3.12 -m venv .report-venv
.report-venv/bin/python -m pip install -r requirements-report.txt
.report-venv/bin/python scripts/reproduce_paper.py
```

This reads the included scientific export and creates LaTeX tables, PDF figures, PNG previews and verification records in the ignored `results/generated/` directory. No GPU or model weights are needed. Use `--tables-only` for tables or `--output-dir PATH` for another destination.

## Repository structure

| Directory | Role |
|---|---|
| `dpnn_experiments/` | Scientific implementation: data, operators, networks, training and evaluation |
| `src/` | U-Net architecture used by the experiment |
| `scripts/` | Configure paths, launch the study and reproduce paper results |
| `configs/` | Study/preparation/test templates and recorded paper settings |
| `tests/` | Numerical correctness, archive integrity and continuation checks |
| `tools/` | Operator-factorization and layer diagnostics |
| `results/data/` | Compact public scientific export: metrics, selected image arrays and scientific metadata |
| `results/paper/` | Paper tables and figures generated from that export |
| `supplement/` | Experimental documentation |

Code defines **how** the experiment works; configurations define **which settings** it uses; scripts provide entry points. Configuration templates contain neutral artifact IDs rather than workstation or server paths. The [path configuration guide](supplement/reproducibility.md#local-path-configuration) explains how to generate local configurations before starting a run.

Local `runs/`, `local/`, `artifacts/` and `results/generated/` are ignored by Git. They are not publication directories. Full operational archives and historical report packages are not included in the public export.

## Artifact scope and citation

The checkout contains enough exported data to reproduce the paper tables and figures, but not all training data, selected model weights or large projection factors. Re-running inference or training requires additional bound artifacts. No public download index is supplied yet.

The recorded study used a 30-epoch phase followed by continuation to 60 epochs. `configs/paper/` documents the completed study; the executable historical starting profile is separate under `configs/study/`. Reorganized source files do not replace the original source snapshots for resuming historical runs.

Please cite **Simon Göppel, Jürgen Frikel, and Markus Haltmeier, “Data-proximal null-space networks for inverse problems”**, and record the exact code commit used.

## License

The program code in this repository is licensed under the **GNU General Public License, version 3 only (SPDX: GPL-3.0-only)**. See [LICENSE](LICENSE) for the full terms. This covers the Python implementation, scripts, tests and tools. The scientific purpose stated above is not a restriction on uses permitted by the license.

[Third-party notices](THIRD_PARTY_NOTICES.md) identify the U-Net reference implementation, observed adaptations and separately installed dependencies. Upstream notices remain applicable. The code license does not relicense external libraries, the accompanying paper, datasets, generated result files or model weights; no separate reuse license for those non-code artifacts is granted here.

The software is provided without warranty to the extent permitted by applicable law. Historical checkpoint source snapshots remain distinct from this release, as explained in the reproducibility guide.
