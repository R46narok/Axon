#ifndef APOLLO_F1_MATH_CUDA_NATIVE_TRANSPOSE_CUH
#define APOLLO_F1_MATH_CUDA_NATIVE_TRANSPOSE_CUH

#include "core.cuh"

F1_EXTERN_BEGIN

F1_EXPORT void F1_API transpose(void* input, void* output, int rows, int columns);

F1_EXTERN_END

#endif //APOLLO_F1_MATH_CUDA_NATIVE_TRANSPOSE_CUH
