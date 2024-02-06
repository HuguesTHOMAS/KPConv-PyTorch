# kpconv_torch

Created by Hugues THOMAS

## Introduction

This repository contains the implementation of **Kernel Point Convolution** (KPConv) in [PyTorch](https://pytorch.org/).

Another implementation of KPConv is available in [PyTorch-Points-3D](https://github.com/nicolas-chaulet/torch-points3d)

## Introduction

KPConv is a point convolution operator presented in the Hugues Thomas's ICCV2019 paper
([arXiv](https://arxiv.org/abs/1904.08889)). Consider citing:

```
@article{thomas2019KPConv,
    Author = {Thomas, Hugues and Qi, Charles R. and Deschaud, Jean-Emmanuel and Marcotegui, Beatriz and Goulette, Fran{\c{c}}ois and Guibas, Leonidas J.},
    Title = {KPConv: Flexible and Deformable Convolution for Point Clouds},
    Journal = {Proceedings of the IEEE International Conference on Computer Vision},
    Year = {2019}
}
```

![Intro figure](./doc/Github_intro.png)

## Installation

This implementation has been tested on Ubuntu 18.04 and Windows 10. Details are provided in [INSTALL.md](./INSTALL.md).

## Experiments

Scripts for three experiments are provided (ModelNet40, S3DIS and SemanticKitti). The instructions to run these experiments are in the [doc](./doc) folder.

:warning: Disclaimer: in this repo version, we only maintain the S3DIS material regarding Scene
Segmentation. Instructions to train KP-FCNN on a scene segmentation task (S3DIS) can be found in
the [doc](./doc/scene_segmentation_guide.md).

As a bonus, a [visualization scripts](./doc/visualization_guide.md) has been implemented: the
kernel deformations display.

## Acknowledgment

Initial tribute to Hugues Thomas, this repo is a fork of [KPConv-PyTorch](https://github.com/HuguesTHOMAS/KPConv-PyTorch) repo.

The code uses the <a href="https://github.com/jlblancoc/nanoflann">nanoflann</a> library.

## License

The code is released under MIT License (see LICENSE file for details).
