#include "core/loss.cuh"
#include "core/compute.cuh"

#include <iostream>
#include <cassert>
#include <cuda_runtime.h>

namespace axon
{
    cross_entropy_loss::cross_entropy_loss()
    {
        cudaMalloc((void**)&pDeviceLoss_, sizeof(float));
    }

    cross_entropy_loss::~cross_entropy_loss()
    {
        if (pDeviceLoss_ != nullptr)
            cudaFree(pDeviceLoss_);

        if (pDeviceWorkspace_ != nullptr)
            cudaFree(pDeviceWorkspace_);
    }

    void cross_entropy_loss::initWorkspace(int batchSize)
    {
        if (pDeviceWorkspace_ == nullptr)
            cudaMalloc((void**)&pDeviceWorkspace_, sizeof(float) * batchSize);
    }

    __device__ float clip(float prediction, float epsilon = 1e-12)
    {
        return fmin(fmax(prediction, epsilon), 1.0f - epsilon);
    }

    __global__ void softmax_loss_kernel(float* pReducedLoss, float* pPredict, float* pTarget, float* pWorkspace, int batchSize, int numOutputs)
    {
        int batch_idx = blockDim.x * blockIdx.x + threadIdx.x;

        extern __shared__ float s_data[];
        float loss = 0.f;

        // each thread calculate entropy for each data and accumulate to shared memory
        for (int c = 0; c < numOutputs; c++)
            loss += pTarget[batch_idx * numOutputs + c] * logf(pPredict[batch_idx * numOutputs + c]);
        pWorkspace[batch_idx] = -loss;

        // then, we do reduction the result to calculate loss using 1 thread block
        if (blockIdx.x > 0) return;

        // accumulate pWorkspace data
        s_data[threadIdx.x] = 0.f;
        for (int i = 0; i < batchSize; i += blockDim.x)
        {
            s_data[threadIdx.x] += pWorkspace[threadIdx.x + i];
        }

        __syncthreads();

        // reduction
        for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1)
        {
            if (threadIdx.x + stride < batchSize)
                s_data[threadIdx.x] += s_data[threadIdx.x + stride];

            __syncthreads();
        }

        if (threadIdx.x == 0)
        {
            pReducedLoss[blockIdx.x] = s_data[0];
        }
    }

    float cross_entropy_loss::loss(blob_f32 *predict, blob_f32 *target)
    {
        int numSms;
        int numBlocksPerSm;

        cudaDeviceGetAttribute(&numSms, cudaDevAttrMultiProcessorCount, 0);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, softmax_loss_kernel, BLOCK_DIM_1D, BLOCK_DIM_1D * sizeof(float));

        int batchSize = target->n();
        int numOutputs = target->c();

        initWorkspace(batchSize);

        #if DEBUG_LOSS
        std::cout << "[[LOSS]]" << std::endl;
        predict->print("predict", true);
        target->print("target", true);
        #endif

        int numBlocks = std::min(numBlocksPerSm * numSms, \
                         (target->size() + BLOCK_DIM_1D - 1) / BLOCK_DIM_1D);

        softmax_loss_kernel<<< numBlocks, BLOCK_DIM_1D, BLOCK_DIM_1D * sizeof(float), 0 >>>
                (pDeviceLoss_, predict->cuda(), target->cuda(), pDeviceWorkspace_, batchSize, numOutputs);
        cudaMemcpy(&hostLoss_, pDeviceLoss_, sizeof(float), cudaMemcpyDeviceToHost);

        // batch mean loss
        return hostLoss_ / float(batchSize);
    }

    float cross_entropy_loss::accuracy(blob_f32 *predict, blob_f32 *target)
    {
        return 0.0f;
    }
}