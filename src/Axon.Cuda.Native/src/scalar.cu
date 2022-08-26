#include "scalar.cuh"
#include "nvtx3/nvToolsExt.h"

__global__ void add_scalar_kernel(float* pInput, float* pOutput, int iLength, float scalar)
{
    for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
         i < iLength;
         i += blockDim.x * gridDim.x)
    {
        pOutput[i] = pInput[i] + scalar;
    }
}

void add_scalar(void* pInput, void* pOutput, int iLength, float scalar)
{
    nvtxRangePush(__FUNCTION__);

    add_scalar_kernel<<<512, 512>>>((float*)pInput, (float*)pOutput, iLength, scalar);

    F1_CUDA_ASSERT(cudaPeekAtLastError());
    nvtxRangePop();
}

__global__ void subtract_scalar_kernel(float* pInput, float* pOutput, int iLength, float scalar)
{
    for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
         i < iLength;
         i += blockDim.x * gridDim.x)
    {
       pOutput[i] = pInput[i] - scalar;
    }
}

void subtract_scalar(void* pInput, void* pOutput, int iLength, float scalar)
{
    nvtxRangePush(__FUNCTION__);
    subtract_scalar_kernel<<<512, 512>>>((float*)pInput, (float*)pOutput, iLength, scalar);
    F1_CUDA_ASSERT(cudaPeekAtLastError());
    nvtxRangePop();
}
