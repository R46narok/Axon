#ifndef APOLLO_F1_MATH_CUDA_NATIVE_POINTWISE_OPERATIONS_CUH
#define APOLLO_F1_MATH_CUDA_NATIVE_POINTWISE_OPERATIONS_CUH

#include "core.cuh"

F1_EXTERN_BEGIN

/// Performs elementwise addition on two matrices or two vectors
/// \param pFirst A valid pointer to the first operand
/// \param pSecond A valid pointer to the second operand
/// \param pOutput A large enough storage to store the output
/// \param iLength Element count in the sequences
F1_EXPORT void F1_API pointwise_addition(void* pFirst, void* pSecond, void* pOutput, int iLength);

/// Performs elementwise subtraction on two matrices or two vectors
/// \param pFirst A valid pointer to the first operand
/// \param pSecond A valid pointer to the second operand
/// \param pOutput A large enough storage to store the output
/// \param iLength Element count in the sequences
F1_EXPORT void F1_API pointwise_subtraction(void* pFirst, void* pSecond, void* pOutput, int iLength);

/// Performs elementwise multiplication on two matrices or two vectors
/// \param pFirst A valid pointer to the first operand
/// \param pSecond A valid pointer to the second operand
/// \param pOutput A large enough storage to store the output
/// \param iLength Element count in the sequences
F1_EXPORT void F1_API pointwise_multiplication(void* pFirst, void* pSecond, void* pOutput, int iLength);

/// Computes the logarithm of each element of a sequence
/// \param pInput A valid pointer to the sequence
/// \param pOutput A large enough storage to hold the result
/// \param iLength Element count in the sequence
F1_EXPORT void F1_API pointwise_log(void* pInput, void* pOutput, int iLength);

/// Performs elementwise scaled subtraction on two matrices or two vectors
/// \param pFirst A valid pointer to the first operand
/// \param pSecond A valid pointer to the second operand
/// \param pOutput A large enough storage to store the output
/// \param iLength Element count in the sequences
/// \param scale For the second operand
F1_EXPORT void F1_API pointwise_scaled_subtraction(void* pFirst, void* pSecond, void* pOutput, int iLength, float scale);

F1_EXTERN_END

#endif //APOLLO_F1_MATH_CUDA_NATIVE_POINTWISE_OPERATIONS_CUH
