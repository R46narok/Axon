#include "Core/Matrix.cuh"
#include "Kernels/PointwiseKernels.cuh"
#include "Kernels/TransformationKernels.cuh"

#include <iostream>

namespace Axon
{
    Matrix::Matrix(uint32_t rows, uint32_t columns)
        : m_Rows(rows), m_Columns(columns), m_Elements(rows * columns)
    {
    }

    Matrix::~Matrix()
    = default;

    bool Matrix::EqualDimensions(const Matrix& first, const Matrix& second)
    {
        bool eq = first.m_Rows == second.m_Rows && first.m_Columns == second.m_Columns;
        if (!eq) std::cout << "Wrong dimensions" << std::endl;
        return eq;
    }

    bool Matrix::EqualDimensions(const Matrix& first, const Matrix& second, const Matrix& third)
    {
        return EqualDimensions(first, second) && EqualDimensions(first, third);
    }

    bool Matrix::PointwiseAdd(const Matrix &other, const Matrix &output)
    {
        if (!EqualDimensions(*this, other, output)) return false;

        auto pThis = GetDevicePointer();
        auto pOther = other.GetDevicePointer();
        auto pOutput = (float*)output.GetDevicePointer();

        PointwiseMatrixAdditionKernel<<<512, 512>>>(pThis, pOther, pOutput, m_Rows * m_Columns);

        return true;
    }

    bool Matrix::PointwiseMultiply(const Matrix &other, const Matrix &output)
    {
        if (!EqualDimensions(*this, other, output)) return false;

        auto pThis = GetDevicePointer();
        auto pOther = other.GetDevicePointer();
        auto pOutput = (float*)output.GetDevicePointer();

        PointwiseMatrixMultiplicationKernel<<<512, 512>>>(pThis, pOther, pOutput, m_Rows * m_Columns);

        return true;
    }

    bool Matrix::PointwiseSubtract(const Matrix &other, const Matrix &output)
    {
        if (!EqualDimensions(*this, other, output)) return false;

        auto pThis = GetDevicePointer();
        auto pOther = other.GetDevicePointer();
        auto pOutput = (float*)output.GetDevicePointer();

        PointwiseMatrixSubtractionKernel<<<512, 512>>>(pThis, pOther, pOutput, m_Rows * m_Columns);

        return true;
    }

    bool Matrix::ScalarAdd(float scalar, const Matrix &output)
    {
        if (!EqualDimensions(*this, output)) return false;

        auto pThis = GetDevicePointer();
        auto pOutput = (float*)output.GetDevicePointer();

        ScalarMatrixAdditionKernel<<<512, 512>>>(pThis, pOutput, scalar, m_Rows * m_Columns);

        return true;
    }

    bool Matrix::ScalarMultiply(float scalar, const Matrix &output)
    {
        if (!EqualDimensions(*this, output)) return false;

        auto pThis = GetDevicePointer();
        auto pOutput = (float*)output.GetDevicePointer();

        ScalarMatrixMultiplicationKernel<<<512, 512>>>(pThis, pOutput, scalar, m_Rows * m_Columns);

        return true;
    }

    bool Matrix::ScalarSubtract(float scalar, const Matrix &output)
    {
        if (!EqualDimensions(*this, output)) return false;

        auto pThis = GetDevicePointer();
        auto pOutput = (float*)output.GetDevicePointer();

        ScalarMatrixSubtractionKernel<<<512, 512>>>(pThis, pOutput, scalar, m_Rows * m_Columns);

        return true;
    }

    MatrixOperation Matrix::operator+(const Matrix &other)
    {
        MatrixOperation operation{};
        operation.pFirst = this;
        operation.pSecond = const_cast<Matrix *>(&other);
        operation.flags = MatrixOperation::MatrixAddition;

        return operation;
    }

    void Matrix::operator=(const MatrixOperation& operation)
    {
        switch (operation.flags)
        {
            case MatrixOperation::MatrixAddition: operation.pFirst->PointwiseAdd(*operation.pSecond, *this); break;
            case MatrixOperation::MatrixSubtraction: operation.pFirst->PointwiseSubtract(*operation.pSecond, *this);break;
            case MatrixOperation::MatrixMultiplication: operation.pFirst->PointwiseMultiply(*operation.pSecond, *this);break;
            case MatrixOperation::MatrixScalarAddition: operation.pFirst->ScalarAdd(operation.scalar, *this);break;
            case MatrixOperation::MatrixScalarSubtraction: operation.pFirst->ScalarSubtract(operation.scalar, *this);break;
            case MatrixOperation::MatrixScalarMultiplication: operation.pFirst->ScalarMultiply(operation.scalar, *this);break;
            case MatrixOperation::Dot: operation.pFirst->DotImpl(*operation.pSecond, *this);break;
            case MatrixOperation::Log: operation.pFirst->LogImpl(*this);break;
            case MatrixOperation::Transpose: operation.pFirst->TransposeImpl(*this);break;
            case MatrixOperation::InsertColumn: operation.pFirst->InsertColumnImpl(*this, operation.scalar);break;
            case MatrixOperation::RemoveColumn: operation.pFirst->RemoveColumnImpl(*this, (int)operation.scalar);break;
            default: break;
        }
    }

    MatrixOperation Matrix::operator-(const Matrix &other)
    {
        MatrixOperation operation{};
        operation.pFirst = this;
        operation.pSecond = const_cast<Matrix *>(&other);
        operation.flags = MatrixOperation::MatrixSubtraction;

        return operation;
    }

    MatrixOperation Matrix::operator*(const Matrix &other)
    {
        MatrixOperation operation{};
        operation.pFirst = this;
        operation.pSecond = const_cast<Matrix *>(&other);
        operation.flags = MatrixOperation::MatrixMultiplication;

        return operation;
    }

    MatrixOperation Matrix::operator+(float scalar)
    {
        MatrixOperation operation{};
        operation.pFirst = this;
        operation.scalar = scalar;
        operation.flags = MatrixOperation::MatrixScalarAddition;

        return operation;
    }

    MatrixOperation Matrix::operator-(float scalar)
    {
        MatrixOperation operation{};
        operation.pFirst = this;
        operation.scalar = scalar;
        operation.flags = MatrixOperation::MatrixScalarSubtraction;

        return operation;
    }

    MatrixOperation Matrix::operator*(float scalar)
    {
        MatrixOperation operation{};
        operation.pFirst = this;
        operation.scalar = scalar;
        operation.flags = MatrixOperation::MatrixScalarMultiplication;

        return operation;
    }

    MatrixOperation Matrix::Dot(const Matrix &other)
    {
        MatrixOperation operation{};
        operation.pFirst = this;
        operation.pSecond = const_cast<Matrix *>(&other);
        operation.flags = MatrixOperation::Dot;

        return operation;
    }

    void Matrix::DotImpl(const Matrix& other, const Matrix& output)
    {
        auto pThis = GetDevicePointer();
        auto pOther = (float*)other.GetDevicePointer();
        auto pOutput = (float*)output.GetDevicePointer();

        MatrixDotKernel(pThis, pOther, pOutput, m_Rows, m_Columns, other.m_Columns);
    }

    MatrixOperation Matrix::Transpose()
    {
        MatrixOperation operation{};
        operation.pFirst = this;
        operation.flags = MatrixOperation::Transpose;

        return operation;
    }

    void Matrix::TransposeImpl(const Matrix &output)
    {
        auto pThis = GetDevicePointer();
        auto pOutput = (float*)output.GetDevicePointer();

        constexpr int BLOCK_DIM = 8;
        dim3 grid(ceil((float)m_Rows / BLOCK_DIM), ceil((float)m_Columns / BLOCK_DIM), 1);
        dim3 threads(BLOCK_DIM, BLOCK_DIM, 1);
        MatrixTransposeKernel<<<grid, threads>>>(pThis, pOutput, m_Rows, m_Columns);
    }

    MatrixOperation Matrix::InsertColumn(float value)
    {
        MatrixOperation operation{};
        operation.pFirst = this;
        operation.scalar = value;
        operation.flags = MatrixOperation::InsertColumn;

        return operation;
    }

    MatrixOperation Matrix::RemoveColumn(int idx)
    {
        MatrixOperation operation{};
        operation.pFirst = this;
        operation.scalar = (float)idx;
        operation.flags = MatrixOperation::RemoveColumn;

        return operation;
    }

    void Matrix::InsertColumnImpl(const Matrix& output, float value)
    {
        auto pThis = GetDevicePointer();
        auto pOutput = (float*)output.GetDevicePointer();
        MatrixInsertColumnKernel<<<512, 512>>>(pOutput, pThis, m_Rows, m_Columns, value);
    }

    void Matrix::RemoveColumnImpl(const Matrix& output, int idx)
    {
        auto pThis = GetDevicePointer();
        auto pOutput = (float*)output.GetDevicePointer();
        MatrixRemoveColumnKernel<<<512, 512>>>(pOutput, pThis, m_Rows, m_Columns);
    }

    MatrixOperation Matrix::Log()
    {
        MatrixOperation operation{};
        operation.pFirst = this;
        operation.flags = MatrixOperation::Log;
        return operation;
    }

    void Matrix::LogImpl(const Matrix &output)
    {
        auto pThis = GetDevicePointer();
        auto pOutput = (float*)output.GetDevicePointer();
        MatrixLogKernel<<<512, 512>>>(pThis, pOutput, m_Rows * m_Columns);
    }
}