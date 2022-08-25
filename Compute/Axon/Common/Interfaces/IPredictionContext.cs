using Axon.Common.LinearAlgebra;

namespace Axon.Common.Interfaces;

public interface IPredictionContext
{
    public void AllocateMemoryForPredictionBatch(MatrixStorage[] parameters, int batchSize);
}