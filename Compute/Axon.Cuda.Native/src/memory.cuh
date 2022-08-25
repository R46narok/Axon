#ifndef APOLLO_F1_MATH_CUDA_NATIVE_MEMORY_CUH
#define APOLLO_F1_MATH_CUDA_NATIVE_MEMORY_CUH

#include "core.cuh"
#include <stdint.h>

F1_EXTERN_BEGIN

/// Host instruction to allocate global memory on the CUDA device. Undefined
/// behaviours in out of memory situations.
/// \param iBytes Byte width of the buffer
/// \return A pointer to the allocated memory on the VRAM
F1_EXPORT void* F1_API allocate_global_memory(int64_t iBytes);

/// Host instruction to free memory from the VRAM
/// \param ptr A valid pointer to GPU memory
F1_EXPORT void F1_API destroy_global_memory(void* ptr);

/// Host instruction to copy memory from CPU-access area to GPU-access area
/// \param pSrc A valid pointer to a RAM block to copy from
/// \param pDst A valid pointer to a VRAM block to copy to
/// \param iLength Bytes to copy
F1_EXPORT void F1_API copy_host_to_device(void* pSrc, void* pDst, int64_t iLength);

/// Host instruction to copy memory from GPU-access area to CPU-access area
/// \param pSrc A valid pointer to a VRAM block to copy from
/// \param pDst A valid pointer to a RAM block to copy to
/// \param iLength Bytes to copy
F1_EXPORT void F1_API copy_device_to_host(void* pSrc, void* pDst, int64_t iLength);

/// Host instruction to copy memory from GPU-access area to GPU-access area
/// \param pSrc A valid pointer to a VRAM block to copy from
/// \param dst A valid pointer to a VRAM block to copy to
/// \param iLength Bytes to copy
F1_EXPORT void F1_API copy_device_to_device(void* pSrc, void* pDst, int64_t iLength);

/// Host instruction every byte on a GPU-access area to a specific value
/// \param pDst A valid pointer to a VRAM block
/// \param iLength Bytes to set
/// \param value Value to be set
F1_EXPORT void device_memset(void* pDst, int64_t iLength, int value);

F1_EXTERN_END

#endif //APOLLO_F1_MATH_CUDA_NATIVE_MEMORY_CUH
