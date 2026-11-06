# Experimental setup

This document describes the completed 60-epoch study. Effective metadata, rather than development templates, are the source of truth. Internal names `DP` and `DP_TWO` correspond to paper labels DP1 and DP2.

## Data and coordinates

| Quantity | Setting |
|---|---|
| Image domain and size | [-1,1]²; 256 × 256 pixels; h = 2/256 |
| Projection angles | 120 equally spaced angles in [-60°,60°), upper endpoint excluded |
| Detectors | 363 positions in [-√2,√2] |
| Operator | Pixel/ray intersection lengths, full square image domain |
| Image coordinates | Physical pixel values for networks; orthonormal coefficients are h times pixel values |
| Data coordinates | Samples multiplied by √w, with w = angular spacing × detector spacing |
| Phantoms | 3–12 nonnegative ellipses contained in radius 0.9 |
| Major semiaxis; axis ratio | [0.06,0.6]; [0.2,1] |
| Amplitudes | Symmetric Dirichlet parameter 1; sum scaled to a random value in [0.6,1] |
| Pixel targets | 4 × 4 midpoint quadrature per pixel |
| Clean measurements | Analytic ellipse line integrals, independent of the reconstruction matrix |
| Noise | Gaussian direction normalized to 5% of each weighted clean-data norm |
| Splits | 2,500 / 500 / 500, assigned by object before noise generation |

Ellipse centres satisfy a conservative containment condition: centre norm plus major semiaxis does not exceed 0.9. This is not uniform sampling of every possible ellipse placement. A fixed noise realization is used for each object. Geometry seed is 110101 and noise seed 110102; the dataset seed is 20261101. Split identifiers are recorded in `data/metadata/split_ids.json` within the completed-study export.

The normalized matrix is A = (√w/h)L. In pixel coordinates a network correction a has data seminorm √w ||La||₂ = ||A(ha)||₂. The continuous full-space Radon model, the discrete operator, and the finite matrix kernel are distinguished in the paper.

## Initial reconstructions and projections

| Setting | Tikhonov | Isotropic TV |
|---|---|---|
| Regularization parameter | 0.01 | 0.001 |
| Solver | CG on normal equations | Chambolle–Pock |
| Relative tolerance | 1e-8 | 1e-5 |
| Absolute tolerance | 1e-10 | 1e-9 |
| Maximum iterations | 500 | 200,000 |
| Check interval; step safety | — | 50; 0.99 |

Regularization parameters were chosen by visual inspection, as stated by the authors, and fixed for all objects and network architectures. TV uses unscaled forward differences with zero terminal differences and no positivity constraint. Starts are cached and shared across methods. Tolerances refer to the respective implemented solver checks, not a uniform image-error threshold.

The numerical row-space basis has shape 65536 × 38597. Two blockwise SVDs use a block-local relative threshold of 1e-12, followed by stabilization. P̃ = I − QQᵀ is fixed during training. This cutoff is not a global truncated SVD. Small leakage does not alone establish proximity to the exact kernel projector.

## Networks and optimization

RES uses z+U(z); NSN uses z+P̃U(z). DP1 adds a controlled complementary part of the shared correction, and DP2 uses a second independently parametrized U-Net for that part. DP1 controls the data norm of the unprojected shared correction; DP2 controls the projected second correction. In each case β = δ, fixed per object. Both DP2 branches are optimized jointly.

Each U-Net has five levels, 19 convolutional layers, bilinear upsampling, ReLU blocks and a linear output, without batch normalization or dropout. One U-Net has 17,252,993 trainable parameters; DP2 has 34,505,986.

| Parameter | Value |
|---|---|
| Objective | Mean pixel MSE, with no additional TV, fidelity or SSIM loss |
| Optimizer | Adam, learning rate 1e-4, betas (0.9,0.999), epsilon 1e-8 |
| Weight decay; AMSGrad | 0; false |
| Scheduler | StepLR, factor 0.5 every 10 epochs |
| Batch size; gradient clipping | 2; norm limit 1 |
| Total epochs; steps | 60; 75,000 per model |
| Precision | Float64, no mixed precision, TF32 disabled |
| Training input | Physical intensities; no augmentation or normalization |
| Loader | Shuffle training objects; zero workers; no dropped final batch |
| Compute | One NVIDIA H200 per run; four parallel jobs in the original study |

Seed index 0: model 20261101, sampler 771101, Python 991101, NumPy 881101. Seed index 1: model 20261102, sampler 771102, Python 991102, NumPy 881102. Four architectures × two starts × two seeds yield 16 trainings.

## Selection and evaluation

Validation evaluates all 500 validation objects after each epoch. Separate best-MSE and best-SSIM checkpoints retain the earlier checkpoint on ties. SSIM uses an 11 × 11 Gaussian window, sigma 1.5, population moments, valid windows, range 1 and constants k1=0.01/k2=0.03. PSNR also uses intensity range 1.

The original study trained for 30 epochs and continued to 60 while preserving optimizer, scheduler and random states. No original run stopped early; early stopping was disabled for the continuation. Test results from epoch 30 had been inspected before extension. Final checkpoint selection remained validation-based, but final testing reused the same objects and is not independent confirmation.

The paper averages each object's metric across the two training seeds, then reports the mean and sample SD across 500 objects; BASE is included once. Metadata containing population SD must not be copied as paper sample SD.

For output x̂ and start z: d=||A(h(x̂−z))||₂, r=||A(hx̂)−yδ||₂. The budget controls d, not r. With ℓ=||A(hP̃U(z))||₂, the implemented bounds are β+(1−ts)ℓ for DP1 and β+ℓ for DP2. Violations use tolerance 1e-12 + 1e-10 max(data norm of output, data norm of start, adjusted budget). Numerical null/complement errors plus the cross term sum exactly algebraically to pixel MSE; the cross term vanishes for an exact orthogonal projector.
