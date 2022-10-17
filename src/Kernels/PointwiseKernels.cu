//
// Created by Acer on 17.10.2022 г..
//
#include "PointwiseKernels.cuh"

namespace Axon
{
    __global__ void PointwiseAdditionKernel(const float* pFirst, const float* pSecond, float* pOutput, uint32_t length)
    {
        for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
             i < length;
             i += blockDim.x * gridDim.x)
        {
            pOutput[i] = pFirst[i] + pSecond[i];
        }
    }
}