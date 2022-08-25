//
// Created by Acer on 2.7.2022 г..
//

#include "transpose.cuh"


#include "transpose.cuh"

#define BLOCK_DIM 8

__global__ void transpose_kernel(float *odata, float* idata, int width, int height)
{
    unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;

    if (xIndex < width && yIndex < height)
    {
//        unsigned int index_in = yIndex + height * xIndex;
//        unsigned int index_out  = xIndex + width * yIndex;
        unsigned int index_in = yIndex * width + xIndex;
        unsigned int index_out = xIndex * height + yIndex;
        odata[index_out] = idata[index_in];
    }
}

void transpose(void* input, void* output, int rows, int columns)
{
    dim3 grid(ceil((float)rows / BLOCK_DIM), ceil((float)columns / BLOCK_DIM), 1);
    dim3 threads(BLOCK_DIM, BLOCK_DIM, 1);

    transpose_kernel<<< grid, threads >>>((float*)output, (float*)input, rows, columns);
    F1_CUDA_ASSERT(cudaPeekAtLastError());
}
