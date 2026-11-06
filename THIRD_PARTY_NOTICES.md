# Licensing and third-party notices

The repository program code is released under GNU GPL version 3 only (GPL-3.0-only); see [LICENSE](LICENSE). This file records upstream attribution, observed modifications and dependency terms. It supplements the code license and does not replace upstream notices or relicense separately installed software. Non-code artifacts are outside this code-license grant as described in the README.

## U-Net implementation

The architecture originates from Olaf Ronneberger, Philipp Fischer and Thomas Brox, *U-Net: Convolutional Networks for Biomedical Image Segmentation*, MICCAI 2015. The [authors' project page](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/) provides the original Caffe/MATLAB release. An architecture citation does not replace attribution or licensing of a particular implementation.

`src/unet.py` closely matches the following independent PyTorch implementation:

- Project: [milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet).
- Compared revision: `21d7850f2af30a9695bbeea75f3136aa538cfc4a`.
- Compared files: [unet/unet_parts.py](https://github.com/milesial/Pytorch-UNet/blob/21d7850f2af30a9695bbeea75f3136aa538cfc4a/unet/unet_parts.py) and [unet/unet_model.py](https://github.com/milesial/Pytorch-UNet/blob/21d7850f2af30a9695bbeea75f3136aa538cfc4a/unet/unet_model.py).
- Upstream license: **GNU General Public License, version 3**, as supplied in the [upstream LICENSE](https://github.com/milesial/Pytorch-UNet/blob/21d7850f2af30a9695bbeea75f3136aa538cfc4a/LICENSE). An unchanged reference copy is included at [LICENSES/Pytorch-UNet-GPL-3.0.txt](LICENSES/Pytorch-UNet-GPL-3.0.txt).

The correspondence includes the `DoubleConv`, `Down`, `Up`, `OutConv` and `UNet` classes, the bilinear/transposed-convolution alternatives, asymmetric padding, skip concatenation, and the channel/factor arrangement. Relative to the compared upstream revision, the local implementation omits batch normalization and checkpointing support, changes some names and defaults, and combines the components in one file. These are observed differences, not a verified account of when or by whom adaptations were made.

The comparison was performed on 6 November 2026. The original transfer or derivation of this file has not been confirmed. Substantial source similarity is documented; independent authorship or a different permissive license is not asserted. For this release, the local U-Net is distributed under GPL version 3 as part of the GPL-covered program, retaining this upstream attribution. This licensing choice does not establish the original transfer history.

The local adaptations listed above were documented on 6 November 2026; their original dates and authors have not been established. Credit for the compared implementation belongs to the upstream project's authors and contributors. The Free Software Foundation copyright in the GPL document refers to the license text, not authorship of this U-Net implementation.

The release retains the GPL terms for the combined program and provides its source. No additional permission to combine third-party GPL code with otherwise incompatible libraries is asserted. In particular, compatibility of a particular CUDA/cuSOLVER distribution remains subject to the applicable licenses; this notice does not declare a system-library exception or grant an exception on behalf of upstream authors.

## Separately installed dependencies

The source checkout imports or loads the following packages. Their implementations and binary distributions are not bundled here. Version numbers below identify the license texts reviewed; they do not constitute a complete environment lock or imply that every workflow uses every package.

| Dependency | Use in this repository | Reviewed upstream license |
| --- | --- | --- |
| NumPy 1.26.4 | Arrays, synthetic data, operators and reports | [BSD-3-Clause](https://github.com/numpy/numpy/blob/v1.26.4/LICENSE.txt) |
| PyTorch 2.7.0 | Networks, training, differentiation and GPU linear algebra | [BSD-style license; retain the complete upstream notices](https://github.com/pytorch/pytorch/blob/v2.7.0/LICENSE) |
| SciPy 1.11.4 | Independent bootstrap comparison in `tests/study_qualification.py` | [BSD-3-Clause](https://github.com/scipy/scipy/blob/v1.11.4/LICENSE.txt) |
| Matplotlib 3.10.7 | Paper figures | [Matplotlib license, with its historical license notices](https://github.com/matplotlib/matplotlib/blob/v3.10.7/LICENSE/LICENSE) |
| Pillow/PIL 13.0.0, observed in the local report environment | Image output in `dpnn_experiments/study_report.py` | [MIT-CMU](https://github.com/python-pillow/Pillow/blob/13.0.0/LICENSE); pinned in `requirements-training.txt`; this does not establish the version of every historical run |
| NVIDIA CUDA/cuSOLVER | Native GPU factorization and CUDA execution | [NVIDIA CUDA license terms](https://docs.nvidia.com/cuda/eula/index.html); use the terms applicable to the installed/distributed version |

Keep the relevant upstream copyright and license notices when redistributing these packages or their components. A future binary, container or environment bundle requires its own dependency and redistribution review, including transitive libraries; this source-level inventory is not a complete notice set for such a bundle. Loading cuSOLVER does not mean that NVIDIA library source code is included in this repository.

## Other code and review limits

The current 57 Python files were screened for source/provenance notices and external imports. The numerical code includes standard LSQR, conjugate-gradient, TV, SSIM and bootstrap formulas. The local LSQR and SSIM implementations were also compared with SciPy 1.11.4 and `VainF/pytorch-msssim` 1.0.0, respectively. No additional substantial source correspondence was identified in those targeted comparisons. Mathematical agreement or use of a library API alone does not establish that source code was copied.

This is a targeted provenance review, not a guarantee that every possible upstream source has been found. Earlier ASTRA, TorchRadon, ODL/LoDoPaB and scikit-image-related legacy implementations are absent from the current release tree and its new Git history. This notice does not clear the licensing of separately retained historical archives.
