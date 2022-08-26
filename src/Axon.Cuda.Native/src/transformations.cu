#include "transformations.cuh"
#define BLOCK_SIZE 32

__global__ void insert_column_kernel(float *pOutput, float* pInput, int width, int height, float value)
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

void insert_column(void* pInput, void* pOutput, int iRows, int iColumns, float value)
{
    insert_column_kernel<<< 512, 512 >>>((float*)pOutput, (float*)pInput, iRows, iColumns, value);
    F1_CUDA_ASSERT(cudaPeekAtLastError());
}

__global__ void remove_column_kernel(float* pInput, float* pOutput, int width, int height)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < width; i += blockDim.x * gridDim.x)
    {
        for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < height; j += blockDim.y * gridDim.y)
        {
//            int index_in = i * height + j;
//            int index_out = i * (height - 1) + (j - 1);
            int index_in = j * width + i;
            int index_out = (j - 1) * width + i;
            pOutput[index_out] = pInput[index_in];
        }
    }

}

void remove_column(void* src, void* dst, int iRows, int iColumns)
{
    dim3 grid(ceil((float)iRows / BLOCK_SIZE), ceil((float)iColumns / BLOCK_SIZE), 1);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE, 1);

    remove_column_kernel<<<512, 512>>>((float*) src,(float*)  dst, iRows, iColumns);
    F1_CUDA_ASSERT(cudaPeekAtLastError());
}
