//
// Created by Acer on 19.10.2022 г..
//
#ifndef AXON_ACTIVATION_CUH
#define AXON_ACTIVATION_CUH

#include "Core/Library.cuh"
#include "Core/Matrix.cuh"

namespace Axon
{
    class AXON_API Activation
    {
    public:
        static void Sigmoid(const Matrix& dst, const Matrix& src, bool gradient = false);
        static void ReLu(const Matrix& dst, const Matrix& src, bool gradient = false);

        enum class Function : uint8_t
        {
            Sigmoid, ReLu
        };
    };
}

#endif //AXON_ACTIVATION_CUH
