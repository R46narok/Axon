#ifndef APOLLO_F1_MATH_CUDA_NATIVE_FUNCTIONS_OPERATIONS_CUH
#define APOLLO_F1_MATH_CUDA_NATIVE_FUNCTIONS_OPERATIONS_CUH

#include "core.cuh"

F1_EXTERN_BEGIN

/// Executes a CUDA kernel to apply the sigmoid activation function
/// to a vector or a matrix.
/// \param pElements Numerical sequence on the GPU
/// \param iLength Number of elements in the sequence
F1_EXPORT void F1_API function_sigmoid(void* pInput, void* pOutput, int iLength);

/// Executes a CUDA kernel to apply the gradient of the sigmoid activation function to
/// a vector or a matrix.
/// \param pElements Numerical sequence on the GPU
/// \param iLength Number of elements in the sequence
F1_EXPORT void F1_API function_sigmoid_gradient(void* pInput, void* pOutput, int iLength);

F1_EXTERN_END

#endif //APOLLO_F1_MATH_CUDA_NATIVE_FUNCTIONS_OPERATIONS_CUH
