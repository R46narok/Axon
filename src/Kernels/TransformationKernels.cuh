//
// Created by Acer on 18.10.2022 г..
//
#ifndef AXON_TRANSFORMATIONKERNELS_CUH
#define AXON_TRANSFORMATIONKERNELS_CUH

#include "Core/Library.cuh"

namespace Axon
{
    void MatrixDotKernel(float* pFirst, float* pSecond, float* pOutput, int firstRows, int firstColumns, int secondColumns);
    __global__ void MatrixTransposeKernel(float* input, float* output, int rows, int columns);
    __global__ void MatrixInsertColumnKernel(float *pOutput, float* pInput, int width, int height, float value);
    __global__ void MatrixRemoveColumnKernel(float* pInput, float* pOutput, int width, int height);
}

#endif //AXON_TRANSFORMATIONKERNELS_CUH
