#ifndef AXON_MATRIX_CUH
#define AXON_MATRIX_CUH

#include "Core/Interop.cuh"

#include <cstdint>
#include <thrust/device_vector.h>

namespace Axon
{
    class AXON_API Matrix
    {
    public:
        Matrix(uint32_t rows, uint32_t columns);
        ~Matrix() noexcept;

        bool PointwiseAdd(const Matrix& other, const Matrix& output);
        bool PointwiseMultiply(const Matrix& other, const Matrix& output);
        bool PontwiseSubtract(const Matrix& other, const Matrix& output);

        [[nodiscard]] inline thrust::device_vector<float>::iterator Begin() const { return m_Elements.begin(); }
        [[nodiscard]] inline thrust::device_vector<float>::iterator End() const { return m_Elements.end(); }
    private:
        static bool EqualDimensions(const Matrix& first, const Matrix& second);
        static bool EqualDimensions(const Matrix& first, const Matrix& second, const Matrix& third);
    private:
        uint32_t m_Rows;
        uint32_t m_Columns;
        thrust::device_vector<float> m_Elements;
    };
}

#endif //AXON_MATRIX_CUH
