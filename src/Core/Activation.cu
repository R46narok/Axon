//
// Created by Acer on 19.10.2022 г..
//
#include "Core/Activation.cuh"
#include "Kernels/ActivationKernels.cuh"

namespace Axon
{
    void Activation::Sigmoid(const Matrix &dst, const Matrix &src, bool gradient)
    {
        if (!Matrix::EqualDimensions(src, dst)) return;

        auto pInput = src.GetDevicePointer();
        auto pOutput = (float*)dst.GetDevicePointer();

        uint32_t length = src.GetColumns() * src.GetRows();
        if (gradient)
            SigmoidActivationGradientKernel<<<512, 512>>>(pInput, pOutput, length);
        else
            SigmoidActivationKernel<<<512, 512>>>(pInput, pOutput, length);
    }

    void Activation::ReLu(const Matrix &dst, const Matrix &src, bool gradient)
    {

    }
}
