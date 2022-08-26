//
// Created by Acer on 22.7.2022 г..
//

#ifndef APOLLO_F1_COMPUTE_CUDA_NATIVE_RANGE_CUH
#define APOLLO_F1_COMPUTE_CUDA_NATIVE_RANGE_CUH

#include "core.cuh"

F1_EXTERN_BEGIN

F1_EXPORT void F1_API range_push(char* pName);
F1_EXPORT void F1_API range_pop();

F1_EXTERN_END

#endif //APOLLO_F1_COMPUTE_CUDA_NATIVE_RANGE_CUH
