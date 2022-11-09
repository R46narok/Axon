//
// Created by Acer on 18.10.2022 г..
//
#include "TransformationKernels.cuh"
#include <cublas_v2.h>

#pragma comment(lib, "cublas.lib")
#define BLOCK_SIZE 16

namespace Axon
{
    static cublasHandle_t s_Handle;

    void MatrixDotKernel(float* pFirst, float* pSecond, float* pOutput, int firstRows, int firstColumns, int secondColumns)
    {
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
                    pFirst, firstRows,
                    pSecond, firstColumns,
                    &beta,
                    pOutput, firstRows);
    }

    __global__ void MatrixTransposeKernel(float* input, float* output, int rows, int columns)
    {
        unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
        unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;

        if (xIndex < rows && yIndex < columns)
        {
//        unsigned int index_in = yIndex + columns * xIndex;
//        unsigned int index_out  = xIndex + rows * yIndex;
            unsigned int index_in = yIndex * rows + xIndex;
            unsigned int index_out = xIndex * columns + yIndex;
            output[index_out] = input[index_in];
        }
    }


    __global__ void MatrixInsertColumnKernel(float *pOutput, float* pInput, int width, int height, float value)
    {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < width; i += blockDim.x * gridDim.x)
        {
            for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < height; j += blockDim.y * gridDim.y)
            {
//            int index_in = i * height + j;
//            int index_out = i * (height + 1) + (j + 1);
//
//            pOutput[index_out] = pInput[index_in];
//            pOutput[i * (height + 1) + 0] = value;
                int index_in = j * width + i;
                int index_out = (j + 1) * width + i;

                pOutput[index_out] = pInput[index_in];
                pOutput[0 * width + i] = value;
            }
        }
    }

    __global__ void MatrixRemoveColumnKernel(float* pOutput, const float* pInput, uint32_t width, uint32_t height)
    {
        for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < width; i += blockDim.x * gridDim.x)
        {
            for (uint32_t j = blockIdx.y * blockDim.y + threadIdx.y; j < height; j += blockDim.y * gridDim.y)
            {
                if (j != 0)
                {
                    uint32_t index_in = j * width + i;
                    uint32_t index_out = (j - 1) * (width - 1) + i;
                    pOutput[index_out] = pInput[index_in];
                }
           }
        }
    }
}