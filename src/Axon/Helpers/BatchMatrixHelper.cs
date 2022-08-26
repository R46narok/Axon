using System;
using Axon.Common.Buffers;
using Axon.Common.LinearAlgebra;

namespace Axon.Helpers;

public static class BatchMatrixHelper
{
    public static void InitializeBatchAsMatrixArray(IBufferAllocator allocator, out BufferBatch batch, out MatrixStorage[] matrices,
               int length, Func<int, int> rowFunction, Func<int, int> columnFunction,
               BufferDataType dataType, string name)
    {
        var batchElements = new BufferBatchElement[length];
        for (int i = 0; i < length; ++i)
            batchElements[i] = new BufferBatchElement(sizeof(float) * rowFunction(i) * columnFunction(i), dataType, name);
    
        batch = new BufferBatch(allocator, batchElements);
        matrices = new MatrixStorage[length];
    
        for (int i = 0; i < length; ++i)
            matrices[i] = new MatrixStorage(batch[i], rowFunction(i), columnFunction(i), allocator);
    }
}