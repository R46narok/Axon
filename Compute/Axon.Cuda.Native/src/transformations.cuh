#ifndef APOLLO_F1_MATH_CUDA_NATIVE_TRANSFORMATIONS_CUH
#define APOLLO_F1_MATH_CUDA_NATIVE_TRANSFORMATIONS_CUH

#include "core.cuh"

F1_EXTERN_BEGIN

/// Inserts a column at index 0 in a matrix or vector
/// \param pInput A valid pointer to the sequence
/// \param pOutput A large enough storage to hold the result
/// \param iRows
/// \param iColumns
/// \param value Initialization value for the newly inserted column
F1_EXPORT void F1_API insert_column(void* pInput, void* pOutput, int iRows, int iColumns, float value);

/// Inserts a row at index 0 in a matrix or vector
/// \param pInput A valid pointer to the sequence
/// \param pOutput A large enough storage to hold the result
/// \param iRows
/// \param iColumns
/// \param value Initialization value for the newly inserted row
F1_EXPORT void F1_API insert_row(void* pInput, void* pOutput, int iRows, int iColumns, float value);

/// Removes a column at index 0 in a matrix or vector
/// \param pInput A valid pointer to the sequence
/// \param pOutput A large enough storage to hold the result
/// \param iRows
/// \param iColumns
F1_EXPORT void F1_API remove_column(void* pInput, void* pOutput, int iRows, int iColumns);

F1_EXTERN_END

#endif //APOLLO_F1_MATH_CUDA_NATIVE_TRANSFORMATIONS_CUH
