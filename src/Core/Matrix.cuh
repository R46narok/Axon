#ifndef AXON_MATRIX_CUH
#define AXON_MATRIX_CUH

#include "Core/Interop.cuh"

#include <cstdint>
#include <thrust/device_vector.h>

namespace Axon
{
    struct MatrixOperation;

    class AXON_API Matrix
    {
    public:
        Matrix() = default;
        Matrix(uint32_t rows, uint32_t columns);
        ~Matrix() noexcept;

        bool ScalarAdd(float scalar, const Matrix& output);
        bool ScalarMultiply(float scalar, const Matrix& output);
        bool ScalarSubtract(float scalar, const Matrix& output);

        bool PointwiseAdd(const Matrix& other, const Matrix& output);
        bool PointwiseMultiply(const Matrix& other, const Matrix& output);
        bool PointwiseSubtract(const Matrix& other, const Matrix& output);

        MatrixOperation Dot(const Matrix& other);
        MatrixOperation Transpose();
        MatrixOperation Log();

        MatrixOperation InsertColumn(float value);
        MatrixOperation RemoveColumn(int idx);

        void operator=(const MatrixOperation& operation);

        MatrixOperation operator+(const Matrix& other);
        MatrixOperation operator-(const Matrix& other);
        MatrixOperation operator*(const Matrix& other);

        MatrixOperation operator+(float scalar);
        MatrixOperation operator-(float scalar);
        MatrixOperation operator*(float scalar);

        [[nodiscard]] inline thrust::device_vector<float>::const_iterator Begin() const { return m_Elements.begin(); }
        [[nodiscard]] inline thrust::device_vector<float>::const_iterator End() const { return m_Elements.end(); }

        [[nodiscard]] inline thrust::device_vector<float>::iterator Begin() { return m_Elements.begin(); }
        [[nodiscard]] inline thrust::device_vector<float>::iterator End() { return m_Elements.end(); }

        [[nodiscard]] inline uint32_t GetRows() const noexcept { return m_Rows; }
        [[nodiscard]] inline uint32_t GetColumns() const noexcept { return m_Columns; }
        [[nodiscard]] inline const float* GetDevicePointer() const noexcept { return thrust::raw_pointer_cast(m_Elements.data()); }
        [[nodiscard]] inline float* GetDevicePointer() noexcept { return thrust::raw_pointer_cast(m_Elements.data()); }

        static constexpr float Bias = 1.0f;
    public:
        static bool EqualDimensions(const Matrix& first, const Matrix& second);
        static bool EqualDimensions(const Matrix& first, const Matrix& second, const Matrix& third);
    private:
        void DotImpl(const Matrix& other, const Matrix& output);
        void LogImpl(const Matrix& output);
        void TransposeImpl(const Matrix& output);
        void InsertColumnImpl(const Matrix& output, float value);
        void RemoveColumnImpl(const Matrix& output, int idx);
    private:
        uint32_t m_Rows;
        uint32_t m_Columns;
        thrust::device_vector<float> m_Elements;
    };

    struct MatrixOperation
    {
        Matrix* pFirst;
        Matrix* pSecond;
        float scalar;
        uint8_t flags;

        enum
        {
            MatrixAddition, MatrixSubtraction, MatrixMultiplication,
            MatrixScalarAddition, MatrixScalarSubtraction, MatrixScalarMultiplication,
            Dot, Transpose, Log,
            InsertColumn, RemoveColumn
        };
    };
}

#endif //AXON_MATRIX_CUH
