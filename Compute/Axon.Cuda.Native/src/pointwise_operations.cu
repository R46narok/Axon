#include "pointwise_operations.cuh"
#include <nvtx3/nvToolsExt.h>

__global__ void pointwise_addition_kernel(float* pFirst, float* pSecond, float* pOutput, int iLength)
{
    for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
         i < iLength;
         i += blockDim.x * gridDim.x)
    {
        pOutput[i] = pFirst[i] + pSecond[i];
    }
}

void pointwise_addition(void* pFirst, void* pSecond, void* pOutput, int iLength)
{
    nvtxRangePush(__FUNCTION__);

    pointwise_addition_kernel<<<512, 512>>>((float *) pFirst, (float *) pSecond, (float *) pOutput,
                                                            iLength);
    F1_CUDA_ASSERT(cudaPeekAtLastError());
    nvtxRangePop();
}

__global__ void pointwise_subtraction_kernel(float* pFirst, float* pSecond, float* pOutput, int iLength)
{
    for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
         i < iLength;
         i += blockDim.x * gridDim.x)
    {
        pOutput[i] = pFirst[i] - pSecond[i];
    }
}

void pointwise_subtraction(void* pFirst, void* pSecond, void* pOutput, int iLength)
{
    nvtxRangePush(__FUNCTION__);

    pointwise_subtraction_kernel<<<512, 512>>>((float *) pFirst, (float *) pSecond, (float *) pOutput,
                                                            iLength);
    F1_CUDA_ASSERT(cudaPeekAtLastError());
    nvtxRangePop();
}

__global__ void pointwise_multiplication_kernel(float* pFirst, float* pSecond, float* pOutput, int iLength)
{
    for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
         i < iLength;
         i += blockDim.x * gridDim.x)
    {
        pOutput[i] = pFirst[i] * pSecond[i];
    }
}

void pointwise_multiplication(void* pFirst, void* pSecond, void* pOutput, int iLength)
{
    nvtxRangePush(__FUNCTION__);

    pointwise_multiplication_kernel<<<512, 512>>>((float *) pFirst, (float *) pSecond, (float *) pOutput,
                                                               iLength);
    F1_CUDA_ASSERT(cudaPeekAtLastError());
    nvtxRangePop();
}

__global__ void pointwise_log_kernel(float* pInput, float* pOutput, int iLength)
{
    for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
         i < iLength;
         i += blockDim.x * gridDim.x)
    {
        pOutput[i] = log(pInput[i]);
    }
}

void pointwise_log(void* pInput, void* pOutput, int iLength)
{
    nvtxRangePush(__FUNCTION__);
    pointwise_log_kernel<<<512, 512>>>((float*)pInput, (float*)pOutput, iLength);
    F1_CUDA_ASSERT(cudaPeekAtLastError());
    nvtxRangePop();
}

__global__ void pointwise_scaled_subtraction_kernel(float* pFirst, float* pSecond, float* pOutput, int iLength, float scale)
{
    for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
         i < iLength;
         i += blockDim.x * gridDim.x)
    {
        pOutput[i] = pFirst[i] - scale * pSecond[i];
    }
}

void pointwise_scaled_subtraction(void* pFirst, void* pSecond, void* pOutput, int iLength, float scale)
{
    nvtxRangePush(__FUNCTION__);

    pointwise_scaled_subtraction_kernel<<<512, 512>>>((float*)pFirst, (float*)pSecond, (float*)pOutput,
                                                                      iLength, scale);
    F1_CUDA_ASSERT(cudaPeekAtLastError());
    nvtxRangePop();
}
