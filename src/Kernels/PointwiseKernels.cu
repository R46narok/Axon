//
// Created by Acer on 17.10.2022 г..
//
#include "PointwiseKernels.cuh"

namespace Axon
{
    __global__ void PointwiseMatrixAdditionKernel(const float* pFirst, const float* pSecond, float* pOutput, uint32_t length)
    {
        for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
             i < length;
             i += blockDim.x * gridDim.x)
        {
            pOutput[i] = pFirst[i] + pSecond[i];
        }
    }

    __global__ void PointwiseMatrixSubtractionKernel(const float* pFirst, const float* pSecond, float* pOutput, uint32_t length)
    {
        for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
             i < length;
             i += blockDim.x * gridDim.x)
        {
            pOutput[i] = pFirst[i] - pSecond[i];
        }
    }


    __global__ void PointwiseMatrixMultiplicationKernel(const float* pFirst, const float* pSecond, float* pOutput, uint32_t length)
    {
        for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
             i < length;
             i += blockDim.x * gridDim.x)
        {
            pOutput[i] = pFirst[i] * pSecond[i];
        }
    }


    __global__ void ScalarMatrixAdditionKernel(const float* pInput, float* pOutput, float scalar, uint32_t length)
    {
        for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
             i < length;
             i += blockDim.x * gridDim.x)
        {
            pOutput[i] = pInput[i] + scalar;
        }
    }

    __global__ void ScalarMatrixMultiplicationKernel(const float* pInput, float* pOutput, float scalar, uint32_t length)
    {
        for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
             i < length;
             i += blockDim.x * gridDim.x)
        {
            pOutput[i] = pInput[i] * scalar;
        }
    }

    __global__ void ScalarMatrixSubtractionKernel(const float* pInput, float* pOutput, float scalar, uint32_t length)
    {
        for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
             i < length;
             i += blockDim.x * gridDim.x)
        {
            pOutput[i] = pInput[i] - scalar;
        }
    }


    __global__ void MatrixLogKernel(const float* pInput, float* pOutput, uint32_t length)
    {
        for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
             i < length;
             i += blockDim.x * gridDim.x)
        {
            pOutput[i] = log(pInput[i]);
        }
    }
}