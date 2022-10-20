//
// Created by Acer on 18.10.2022 г..
//
#include "ActivationKernels.cuh"

namespace Axon
{
    __global__ void SigmoidActivationKernel(const float* pInput, float* pOutput, uint32_t length)
    {
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
             i < length;
             i += blockDim.x * gridDim.x)
            pOutput[i] = 1.0 / (1 + exp(-1.0 * pInput[i]));
    }

    __global__ void SigmoidActivationGradientKernel(const float* pInput, float* pOutput, uint32_t length)
    {
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
             i < length;
             i += blockDim.x * gridDim.x)
        {
            float sigmoid = 1.0f / (1 + (float)exp(-1.0 * pInput[i]));
            pOutput[i] = sigmoid * (1 - sigmoid);
        }
    }
}
