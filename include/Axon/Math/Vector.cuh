//
// Created by r46narok on 25.03.22 г..
//

#ifndef AXON_VECTOR_CUH
#define AXON_VECTOR_CUH

#include "Axon/Types.cuh"

#include <initializer_list>

namespace Axon
{
    class Vector
    {
    public:
        explicit Vector(int size);
        Vector(const std::initializer_list<float>& elements);

        float Transpose(const Vector& other);
    private:
        std::shared_ptr<Buffer> m_DeviceMemory;
        uint32_t m_Size;
    };
}

#endif //AXON_VECTOR_CUH
