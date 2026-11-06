# Experimental results

The main paper uses validation-MSE-selected checkpoints. All values below refer to the completed 60-epoch study. The accompanying export also contains both seeds and SSIM-selected checkpoints.

## Reconstruction quality

| Initialization | Method | Mean MSE | Mean PSNR (dB) | Mean SSIM |
|---|---|---:|---:|---:|
| Tikhonov | BASE | 1.2125e-3 | 30.63 | 0.5322 |
| Tikhonov | RES | 3.62e-5 | 46.70 | 0.9896 |
| Tikhonov | NSN | 7.590e-4 | 32.95 | 0.5422 |
| Tikhonov | DP1 | 3.69e-5 | 46.54 | 0.9891 |
| Tikhonov | DP2 | 4.43e-5 | 45.46 | 0.9523 |
| TV | BASE | 5.492e-4 | 33.29 | 0.8697 |
| TV | RES | 1.441e-4 | 39.37 | 0.9646 |
| TV | NSN | 3.916e-4 | 34.51 | 0.8184 |
| TV | DP1 | 1.754e-4 | 38.23 | 0.9266 |
| TV | DP2 | 1.902e-4 | 37.88 | 0.8939 |

Rounded means from the paper tables. Objectwise sample SDs and the seed-aggregation rule are documented in the paper; full-precision object metrics are in `data/test/individual_values.csv.gz` in the completed-study export. Both DP variants improve all three mean metrics over NSN for both starts. Tikhonov-DP1 is close to RES; this is not a statistical equivalence claim. TV is the better baseline, but Tikhonov produces the better DP1 reconstruction. DP2 has no measured quality advantage over DP1 in this protocol.

## Validation and checkpoint selection

Validation trajectories summarize the same 500 validation objects for each training seed, not one image. SSIM selection retains the qualitative ranking, while individual metrics change. For TV-NSN the mean test SSIM changes from about 0.818 to 0.849 and mean MSE from 3.92e-4 to 5.09e-4. Falling SSIM alongside improving MSE does not mean that null-space learning has failed; the objectives measure different aspects of quality. Curve/checkpoint exports should be read together, not used to select whichever test result looks best.

## Data control

DP1 reduces its correction for 2/2 of 500 Tikhonov objects (seeds 0/1) and 258/292 TV objects. DP2 reduces all 500/500 corrections for both starts. No DP bound violations exceed the specified tolerance for either selection rule. RES exceeds the noise-norm comparison threshold on 2/3 Tikhonov objects and 367/385 TV objects; the largest normalized change is 9.562. This threshold is not a RES guarantee. The ECDF plots show the fraction of objects at or below each normalized data change.

NSN changes predicted data only at numerical precision. Data proximity does not force the total measurement residual to decrease. Mean discretization-error/noise and effective-error/noise ratios are 0.152 and 1.015.

## Error components and images

For Tikhonov, NSN reduces mean null-space MSE from 4.78e-4 to 2.44e-5 but retains complementary MSE near 7.35e-4; DP1 reduces the latter to 1.17e-5. With TV, complementary MSE falls from approximately 2.96e-4 to 9.49e-5 under DP1. These components may contain both propagated noise and baseline bias; they do not directly measure continuous wavefront sets.

The main paper example is test object 000002, selected for spatial detail rather than best method performance. Additional figures should include the second seed, SSIM-selected reconstructions and an explicitly retrospective difficult case. Use common display windows and keep the selection rule visible. The export contains Float64 arrays and their bindings in `data/examples/index.json`; the displayed subset is not the full test dataset.

## Computational cost and optional statistics

DP2 doubles trainable parameters and approximately doubles optimization and inference time. Optimization timing excludes validation/logging; batch-one inference includes projections and clipping but excludes initial reconstruction and model loading. These are observed study timings, not a randomized latency benchmark.

Paired analyses remain optional supporting material in `data/test/paired_effects.json`. BCa denotes bias-corrected and accelerated bootstrap intervals. Resampling objects, after within-object seed averaging where appropriate, estimates object variation conditional on the trained models; it does not quantify variability over a population of training runs. Keep selection rules and seeds explicitly labelled. The final main paper reports descriptive quality comparisons, not these intervals.

## Public result files

The compact export in `results/data/` contains per-object metrics, paired comparisons, validation trajectories, timing summaries and selected Float64 image arrays. `results/data/export.json` records the derivation from the original study exports. Full operational archives and historical report packages are separate from this public scientific subset.

The final-paper [tables](../results/paper/tables/) and [figures](../results/paper/figures/) can be regenerated using the [reproducibility guide](reproducibility.md).
