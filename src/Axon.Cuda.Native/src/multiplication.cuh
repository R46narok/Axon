#ifndef APOLLO_F1_MATH_CUDA_NATIVE_MULTIPLICATION_CUH
#define APOLLO_F1_MATH_CUDA_NATIVE_MULTIPLICATION_CUH

#include "core.cuh"

F1_EXTERN_BEGIN

/// Performs the dot product on two matrices
/// \param pFirst A valid pointer to the elements of the first matrix on the VRAM
/// \param pSecond A valid pointer to the elements of the second matrix on the VRAM
/// \param pOutput A large enough storage to contain the output of the product
/// \param firstRows
/// \param firstColumns
/// \param secondColumns
F1_EXPORT void F1_API multiply(void* pFirst, void* pSecond, void* pOutput,
                               int firstRows, int firstColumns, int secondColumns);

/// Performs scalar multiplication on a matrix or a vector
/// \param pInput A valid pointer to the elements of the numerical sequence
/// \param pOutput A large enough storage(at least iLength * sizeof(float) bytes long) to contain the output
/// \param iLength Number of elements in the sequence
/// \param scalar Number to multiply each element by
F1_EXPORT void F1_API multiply_scalar(void* pInput, void* pOutput, int iLength, float scalar);

F1_EXTERN_END

#endif //APOLLO_F1_MATH_CUDA_NATIVE_MULTIPLICATION_CUH
