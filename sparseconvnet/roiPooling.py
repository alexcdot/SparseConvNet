# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sparseconvnet.SCN
from torch.autograd import Function
from torch.nn import Module
from .utils import *
from .sparseConvNetTensor import SparseConvNetTensor


class RoiPoolingFunction(Function):
    @staticmethod
    def forward(
            ctx,
            input_features,
            input_metadata,
            input_spatial_size,
            output_spatial_size,
            dimension,
            roi_boxes,
            pool_size,
            pool_stride,
            nFeaturesToDrop):
        ctx.input_metadata = input_metadata
        ctx.dimension = dimension
        ctx.nFeaturesToDrop = nFeaturesToDrop
        output_features = input_features.new()
        sparseconvnet.SCN.RoiPooling_updateOutput(
            input_spatial_size,
            output_spatial_size,
            pool_size,
            input_metadata.getSpatialLocations(input_spatial_size),
            roi_boxes,
            input_metadata,
            input_features,
            output_features,
            nFeaturesToDrop)
        ctx.save_for_backward(
            input_features,
            output_features,
            input_spatial_size,
            output_spatial_size,
            pool_size,
            pool_stride)
        return output_features

    @staticmethod
    def backward(ctx, grad_output):
        input_features,\
            output_features,\
            input_spatial_size,\
            output_spatial_size,\
            pool_size,\
            pool_stride = ctx.saved_tensors
        grad_input = grad_output.new()
        sparseconvnet.SCN.RoiPooling_updateGradInput(
            input_spatial_size,
            output_spatial_size,
            pool_size,
            pool_stride,
            ctx.input_metadata,
            input_features,
            grad_input,
            output_features,
            grad_output,
            ctx.nFeaturesToDrop)
        return grad_input, None, None, None, None, None, None, None


class RoiPooling(Module):
    def __init__(self, dimension, pool_size, pool_stride, out_size, nFeaturesToDrop=0):
        super(RoiPooling, self).__init__()
        self.dimension = dimension
        self.pool_size = toLongTensor(dimension, pool_size)
        self.pool_stride = toLongTensor(dimension, pool_stride)
        self.nFeaturesToDrop = nFeaturesToDrop
        self.out_size = torch.tensor([out_size, out_size])

    def forward(self, input, roi_boxes):
        output = SparseConvNetTensor()
        output.metadata = input.metadata
        output.spatial_size = self.out_size
        output.features = RoiPoolingFunction.apply(
            input.features,
            input.metadata,
            input.spatial_size,
            output.spatial_size,
            self.dimension,
            roi_boxes,
            self.pool_size,
            self.pool_stride,
            self.nFeaturesToDrop)        
        return output

    def __repr__(self):
        s = 'RoiPooling'
        if self.pool_size.max().item() == self.pool_size.min().item() and\
                self.pool_stride.max().item() == self.pool_stride.min().item():
            s = s + str(self.pool_size[0].item()) + \
                '/' + str(self.pool_stride[0].item())
        else:
            s = s + '(' + str(self.pool_size[0].item())
            for i in self.pool_size[1:]:
                s = s + ',' + str(i.item())
            s = s + ')/(' + str(self.pool_stride[0].item())
            for i in self.pool_stride[1:]:
                s = s + ',' + str(i.item())
            s = s + ')'

        if self.nFeaturesToDrop > 0:
            s = s + ' nFeaturesToDrop = ' + self.nFeaturesToDrop
        return s
