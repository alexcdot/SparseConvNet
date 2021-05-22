// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
#include <stdio.h>
#include <iostream>
using namespace std;

template <typename T>
void RoiPooling_ForwardPass(T *input_features, T *output_features, Int nPlanes,
                            Int input_stride, Int output_stride, const Int *rules,
                            Int nHot) {
  Int outSite;
#pragma omp parallel for private(outSite)
  for (outSite = 0; outSite < nHot; outSite++) {
    // cout << "outSite " << outSite <<  ": rules[2 * outSite]: " << rules[2 * outSite] * input_stride << " rules[2 * outSite + 1]: " << rules[2 * outSite + 1] * output_stride << endl;
    Int i = rules[2 * outSite] * input_stride;
    Int o = rules[2 * outSite + 1] * output_stride;
    for (Int plane = 0; plane < nPlanes; plane++)
      if (output_features[o + plane] < input_features[i + plane])
        output_features[o + plane] = input_features[i + plane];
  }
}
template <typename T>
void RoiPooling_BackwardPass(T *input_features, T *d_input_features,
                             T *output_features, T *d_output_features,
                             Int nPlanes, Int input_stride, Int output_stride,
                             const Int *rules, Int nHot) {
  Int outSite;
#pragma omp parallel for private(outSite)
  for (outSite = 0; outSite < nHot; outSite++) {
    Int i = rules[2 * outSite] * input_stride;
    Int o = rules[2 * outSite + 1] * output_stride;
    for (Int plane = 0; plane < nPlanes; plane++)
      if (output_features[o + plane] == input_features[i + plane])
        d_input_features[i + plane] += d_output_features[o + plane];
  }
}

/* inputSize: input_spatial size, outputSize: output_spatial_size (not flatteend)
 */
template <typename T, Int Dimension>
void cpu_RoiPooling_updateOutput(
    /*long*/ at::Tensor &inputSize, /*long*/ at::Tensor &outputSize,
    /*long*/ at::Tensor &poolSize, /*long*/ at::Tensor &inputSpatialLocations,
    /*long*/ at::Tensor &roiBoxes,
    Metadata<Dimension> &m,
    /*float*/ at::Tensor &input_features,
    /*float*/ at::Tensor &output_features, long nFeaturesToDrop) {

  const auto &_rules =
      m.getRuleBook(inputSize, outputSize, poolSize, poolSize, true);

  auto oS = outputSize.data_ptr<long>(); // output shape
  Int nActive = roiBoxes.size(0) * oS[0] * oS[1];  
  output_features.resize_({nActive, input_features.size(1)});
  output_features.zero_();

  auto iF = input_features.data_ptr<T>() + nFeaturesToDrop;
  auto oF = output_features.data_ptr<T>();

  auto rB = roiBoxes.data_ptr<long>();
  auto iLoc = inputSpatialLocations.data_ptr<long>();

  for (int i = 0; i < roiBoxes.size(0); i++) {
    for (int j = 0; j < m.getSpatialLocations(inputSize).size(0); j++) {
      int batchIdx = rB[5 * i], xmin = rB[5 * i + 1], ymin = rB[5 * i + 2], xmax = rB[5 * i + 3], ymax = rB[5 * i + 4];
      if (iLoc[3*j + 2] != batchIdx) {
        continue;
      }
      if (iLoc[3*j] < ymin || iLoc[3*j] > ymax || iLoc[3*j+1] < xmin || iLoc[3*j+1] > xmax) {
        continue;
      }

      // get pool output idx (row-order)
      int poolIdxX = -1, poolIdxY = -1;
      int width = xmax - xmin + 1, height = ymax - ymin + 1;
      int xIdx = 0, xRem = width % oS[0];
      for (int k = 0; k < oS[0]; k++) {
          if (xRem > 0) {
            xIdx += width / oS[0] + 1;
            xRem --;
          } else {
            xIdx += width / oS[0];
          }
          if (iLoc[3*j+1] - xmin <= xIdx - 1) {
            poolIdxX = k;
            break;
          }
      }
      int yIdx = 0, yRem = height % oS[1];
      for (int k = 0; k < oS[1]; k++) {
          if (yRem > 0) {
            yIdx += height / oS[1] + 1;
            yRem --;
          } else {
            yIdx += height / oS[1];
          }
          if (iLoc[3*j] - ymin <= yIdx - 1) {
            poolIdxY = k;
            break;
          }
      }
      
      int poolIdx = batchIdx * oS[0] * oS[1] + poolIdxY * oS[0] + poolIdxX;
      
      if (oF[poolIdx] < iF[j]) {
        oF[poolIdx] = iF[j];
      }
    }
  }
}
template <typename T, Int Dimension>
void cpu_RoiPooling_updateGradInput(
    /*long*/ at::Tensor &inputSize, /*long*/ at::Tensor &outputSize,
    /*long*/ at::Tensor &poolSize,
    /*long*/ at::Tensor &poolStride, Metadata<Dimension> &m,
    /*float*/ at::Tensor &input_features,
    /*float*/ at::Tensor &d_input_features,
    /*float*/ at::Tensor &output_features,
    /*float*/ at::Tensor &d_output_features, long nFeaturesToDrop) {

  Int nPlanes = input_features.size(1) - nFeaturesToDrop;
  const auto &_rules =
      m.getRuleBook(inputSize, outputSize, poolSize, poolStride, true);
  d_input_features.resize_as_(input_features);
  d_input_features.zero_();

  auto iF = input_features.data_ptr<T>();
  auto oF = output_features.data_ptr<T>();
  auto diF = d_input_features.data_ptr<T>();
  auto doF = d_output_features.data_ptr<T>();

  /* number of iterations is equal to the poolSize squared */
  for (auto &r : _rules) {
    Int nHot = r.size() / 2;
    RoiPooling_BackwardPass<T>(iF, diF, oF, doF, nPlanes,
                               input_features.stride(0),
                               output_features.stride(0), &r[0], nHot);
  }
}
