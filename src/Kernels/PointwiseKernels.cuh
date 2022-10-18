//
// Created by Acer on 17.10.2022 г..
//
#ifndef AXON_POINTWISEKERNELS_CUH
#define AXON_POINTWISEKERNELS_CUH

#include <cstdint>
#include "Core/Interop.cuh"

namespace Axon
{
    __global__ void PointwiseMatrixAdditionKernel(const float* pFirst, const float* pSecond, float* pOutput, uint32_t length);
    __global__ void PointwiseMatrixMultiplicationKernel(const float* pFirst, const float* pSecond, float* pOutput, uint32_t length);
    __global__ void PointwiseMatrixSubtractionKernel(const float* pFirst, const float* pSecond, float* pOutput, uint32_t length);

    __global__ void ScalarMatrixAdditionKernel(const float* pInput, float* pOutput, float scalar, uint32_t length);
    __global__ void ScalarMatrixMultiplicationKernel(const float* pInput, float* pOutput, float scalar, uint32_t length);
    __global__ void ScalarMatrixSubtractionKernel(const float* pInput, float* pOutput, float scalar, uint32_t length);
}

#endif //AXON_POINTWISEKERNELS_CUH
