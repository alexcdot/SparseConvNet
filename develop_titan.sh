#!/bin/bash
# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#export TORCH_CUDA_ARCH_LIST="6.0;6.1;6.2;7.0;7.5"
rm -rf build/ dist/ sparseconvnet.egg-info sparseconvnet_SCN*.so
pip3 install --user torch torchvision
python3 setup.py develop --user
# Run max pooling demo (demo is written by us, implementation isn't)
printf "\nRunning max pooling demo\n\n"
python3 examples/max_pooling.py
# Run ROI pooling demo (demo and implementation written by us)
printf "\nRunning ROI pooling demo\n\n"
python3 cpu_demo.py
