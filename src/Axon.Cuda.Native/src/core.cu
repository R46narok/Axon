//
// Created by Acer on 23.7.2022 г..
//

#include "core.cuh"

static ErrorCallback s_Callback;
void set_error_callback(ErrorCallback callback)
{
    s_Callback = callback;
}

void gpuAssert(cudaError_t code, const char* file, int line)
{
    if (code != cudaSuccess && s_Callback != nullptr)
    {
        s_Callback((int)code, cudaGetErrorString(code), file, line);
    }
}
