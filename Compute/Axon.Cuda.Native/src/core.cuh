//
// Created by Acer on 6.7.2022 г..
//

#ifndef APOLLO_F1_MATH_CUDA_NATIVE_CORE_CUH
#define APOLLO_F1_MATH_CUDA_NATIVE_CORE_CUH

#ifdef __cplusplus
#define F1_EXTERN_BEGIN extern "C" {
#define F1_EXTERN_END }
#else
#define F1_EXTERN_BEGIN
#define F1_EXTERN_END
#endif

#define F1_EXPORT __declspec(dllexport)
#define F1_API __cdecl

#include <iostream>
void gpuAssert(cudaError_t code, const char* file, int line);

F1_EXTERN_BEGIN
typedef void (*ErrorCallback)(int, const char*, const char*, int);
F1_EXPORT void F1_API set_error_callback(ErrorCallback callback);
F1_EXTERN_END


#define F1_CUDA_ASSERT(x) gpuAssert((x), __FILE__, __LINE__ );

#endif //APOLLO_F1_MATH_CUDA_NATIVE_CORE_CUH
