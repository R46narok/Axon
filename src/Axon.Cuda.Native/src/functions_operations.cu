#include "functions_operations.cuh"
#include "nvtx3/nvToolsExt.h"
#include <cmath>

__global__ void function_sigmoid_kernel(const float* pInput, float* pOutput, int iLength)
{
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < iLength;
         i += blockDim.x * gridDim.x)
        pOutput[i] = 1.0 / (1 + exp(-1.0 * pInput[i]));
}

void function_sigmoid(void* pInput, void* pOutput, int iLength)
{
    nvtxRangePush(__FUNCTION__);

    function_sigmoid_kernel<<<512, 512>>>((float*)pInput, (float*)pOutput, iLength);
    F1_CUDA_ASSERT(cudaPeekAtLastError());

    nvtxRangePop();
}

__global__ void function_sigmoid_gradient_kernel(const float* pInput, float* pOutput, int iLength)
{
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < iLength;
         i += blockDim.x * gridDim.x)
    {
        float sigmoid = 1.0 / (1 + exp(-1.0 * pInput[i]));
        pOutput[i] = sigmoid * (1 - sigmoid);
    }
}

void function_sigmoid_gradient(void* pInput, void* pOutput, int iLength)
{
    nvtxRangePush(__FUNCTION__);

    function_sigmoid_gradient_kernel<<<512, 512>>>((float*)pInput, (float*)pOutput, iLength);
    F1_CUDA_ASSERT(cudaPeekAtLastError());
    nvtxRangePop();
}
