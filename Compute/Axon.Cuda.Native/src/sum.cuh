#ifndef APOLLO_F1_MATH_CUDA_NATIVE_SUM_CUH
#define APOLLO_F1_MATH_CUDA_NATIVE_SUM_CUH

#include "core.cuh"

F1_EXTERN_BEGIN

/// Sums all elements in a matrix or a vector
/// \param pInput A valid pointer to the elements of the sequence
/// \param pOutput A large enough storage to hold the output (sizeof(float))
/// \param iRows
/// \param iColumns
F1_EXPORT void F1_API sum(void* pInput, void* pOutput, int iRows, int iColumns);

F1_EXTERN_END

#endif //APOLLO_F1_MATH_CUDA_NATIVE_SUM_CUH
