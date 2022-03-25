//
// Created by r46narok on 25.03.22 г..
//

#include "Axon/Math/Vector.cuh"

namespace Axon
{
    Vector::Vector(int size)
        : m_Size(size),
        m_DeviceMemory(Buffer::AllocateDeviceMemory(sizeof(float) * size))
    {

    }

    Vector::Vector(const std::initializer_list<float>& elements)
        : m_Size(elements.size()),
        m_DeviceMemory(Buffer::AllocateDeviceMemory(sizeof(float) * elements.size()))
    {
        cudaMemcpy(m_DeviceMemory->GetDeviceBuffer(), elements.begin(), sizeof(float) * m_Size, cudaMemcpyHostToDevice);
    }

    // TODO: Implement block reduction
    __global__ void TransposeVectorKernel(float* pFirst, float* pSecond, float* pResult, uint32_t size)
    {
        uint32_t i = threadIdx.x + blockDim.x * blockIdx.x;
        if (i < size)
            atomicAdd(pResult, pFirst[i] * pSecond[i]);
    }


    float Vector::Transpose(const Vector &other)
    {
        auto deviceResult = Buffer::AllocateDeviceMemory<float>();
        float hostResult = 0.0f;

        cudaMemcpy(deviceResult->GetDeviceBuffer(), &hostResult, sizeof(float), cudaMemcpyHostToDevice);
        TransposeVectorKernel<<<1, m_Size>>>((float*)m_DeviceMemory->GetDeviceBuffer(),
                                             (float*)other.m_DeviceMemory->GetDeviceBuffer(),
                                             (float*)deviceResult->GetDeviceBuffer(),
                                             m_Size);

        cudaMemcpy(&hostResult, deviceResult->GetDeviceBuffer(), sizeof(float), cudaMemcpyDeviceToHost);

        return hostResult;
    }
}
