//
// Created by Acer on 18.10.2022 г..
//
#ifndef AXON_TRANSFORMATIONKERNELS_CUH
#define AXON_TRANSFORMATIONKERNELS_CUH

#include "Core/Interop.cuh"

namespace Axon
{
    void MatrixDotKernel(float* pFirst, float* pSecond, float* pOutput, int firstRows, int firstColumns, int secondColumns);
    __global__ void MatrixTransposeKernel(float* input, float* output, int rows, int columns);
}

#endif //AXON_TRANSFORMATIONKERNELS_CUH
