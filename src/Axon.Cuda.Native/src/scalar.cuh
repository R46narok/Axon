#ifndef APOLLO_F1_MATH_CUDA_NATIVE_SCALAR_CUH
#define APOLLO_F1_MATH_CUDA_NATIVE_SCALAR_CUH

#include "core.cuh"

F1_EXTERN_BEGIN

F1_EXPORT void F1_API add_scalar(void* pInput, void* pOutput, int iLength, float scalar);
F1_EXPORT void F1_API subtract_scalar(void* pInput, void* pOutput, int iLength, float scalar);

F1_EXTERN_END

#endif //APOLLO_F1_MATH_CUDA_NATIVE_SCALAR_CUH
