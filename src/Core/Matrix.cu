#include "Core/Matrix.cuh"
#include "Kernels/PointwiseKernels.cuh"

namespace Axon
{
    Matrix::Matrix(uint32_t rows, uint32_t columns)
        : m_Rows(rows), m_Columns(columns), m_Elements(rows * columns)
    {
    }

    Matrix::~Matrix()
    = default;

    bool Matrix::PointwiseAdd(const Matrix &other, const Matrix &output)
    {
        if (!EqualDimensions(*this, other, output)) return false;

        auto pThis = thrust::raw_pointer_cast(m_Elements.data());
        auto pOther = thrust::raw_pointer_cast(other.m_Elements.data());
        auto pOutput = (float*)thrust::raw_pointer_cast(output.m_Elements.data());

        PointwiseAdditionKernel<<<512, 512>>>(pThis, pOther, pOutput, m_Rows * m_Columns);
    }

    bool Matrix::EqualDimensions(const Matrix& first, const Matrix& second)
    {
        return first.m_Rows == second.m_Rows && first.m_Columns == second.m_Columns;
    }

    bool Matrix::EqualDimensions(const Matrix& first, const Matrix& second, const Matrix& third)
    {
        return EqualDimensions(first, second) && EqualDimensions(first, third);
    }
}