//
// Created by Acer on 17.10.2022 г..
//
#ifndef AXON_POINTWISEKERNELS_CUH
#define AXON_POINTWISEKERNELS_CUH

#include <cstdint>
#include "Core/Interop.cuh"

namespace Axon
{
    __global__ void PointwiseAdditionKernel(const float* pFirst, const float* pSecond, float* pOutput, uint32_t length);
}

#endif //AXON_POINTWISEKERNELS_CUH
