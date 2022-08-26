#include "memory.cuh"

void* allocate_global_memory(int64_t iBytes)
{
    void* ptr;
    F1_CUDA_ASSERT(cudaMalloc(&ptr, iBytes));
    return ptr;
}

void destroy_global_memory(void* ptr)
{
    cudaFree(ptr);
}

void copy_host_to_device(void* pSrc, void* pDst, int64_t iLength)
{
    cudaMemcpy(pDst, pSrc, iLength, cudaMemcpyHostToDevice);
}

void copy_device_to_host(void* pSrc, void* pDst, int64_t iLength)
{
    cudaMemcpy(pDst, pSrc, iLength, cudaMemcpyDeviceToHost);
}

void copy_device_to_device(void* pSrc, void* pDst, int64_t iLength)
{
    cudaMemcpy(pDst, pSrc, iLength, cudaMemcpyDeviceToDevice);
}

void device_memset(void* pDst, int64_t iLength, int value)
{
    cudaMemset(pDst, value, iLength);
}
