//
// Created by Acer on 18.10.2022 г..
//
#ifndef AXON_ACTIVATIONKERNELS_CUH
#define AXON_ACTIVATIONKERNELS_CUH

#include "Core/Interop.cuh"

namespace Axon
{
    __global__ void SigmoidActivationKernel(const float* pInput, float* pOutput, uint32_t length);
    __global__ void SigmoidActivationGradientKernel(const float* pInput, float* pOutput, uint32_t length);
}

#endif //AXON_ACTIVATIONKERNELS_CUH
