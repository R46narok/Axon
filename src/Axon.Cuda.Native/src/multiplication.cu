#include "multiplication.cuh"
#include "stdio.h"
#include "nvtx3/nvToolsExt.h"
#include <cublas_v2.h>

#pragma comment(lib, "cublas.lib")

#define BLOCK_SIZE 16

__global__ void multiply_kernel(const float* pFirst, const float* pSecond, float* pOutput,
                                int firstRows, int firstColumns, int secondColumns)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < firstRows && col < secondColumns)
    {
        float sum = 0.0;
        int i = 0;
        for (i = 0; i < firstColumns; ++i)
        {
            sum += pFirst[row * firstColumns + i] * pSecond[i * secondColumns + col];
        }
        pOutput[row * secondColumns + col] = sum;
    }
}

static cublasHandle_t s_Handle;

void multiply(void* pFirst, void* pSecond, void* pOutput,
              int firstRows, int firstColumns, int secondColumns)
{
    nvtxRangePush(__FUNCTION__);

    unsigned int grid_rows = ceil((float)firstRows / BLOCK_SIZE);
    unsigned int grid_cols = ceil((float)secondColumns / BLOCK_SIZE);

    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

//    multiply_kernel<<<dimGrid, dimBlock>>>((float*)pFirst, (float*)pSecond, (float*)pOutput,
//                                           firstRows, firstColumns, secondColumns);

    if (s_Handle == nullptr) cublasCreate(&s_Handle);

    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(s_Handle, CUBLAS_OP_N, CUBLAS_OP_N,
                firstRows, secondColumns, firstColumns,
                &alpha,
                (float*)pFirst, firstRows,
                (float*)pSecond, firstColumns,
                &beta,
                (float*)pOutput, firstRows);

    F1_CUDA_ASSERT(cudaPeekAtLastError());
    nvtxRangePop();
}

__global__ void multiply_scalar_kernel(float* pOutput, float* pInput, int iLength, float scalar)
{
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < iLength;
         i += blockDim.x * gridDim.x)
    {
        pOutput[i] = pInput[i] * scalar;
    }
}

void multiply_scalar(void* input, void* pOutput, int iLength, float scalar)
{
    multiply_scalar_kernel<<<512, 512>>>((float *) pOutput, (float *) input, iLength, scalar);
    F1_CUDA_ASSERT(cudaPeekAtLastError());
}
