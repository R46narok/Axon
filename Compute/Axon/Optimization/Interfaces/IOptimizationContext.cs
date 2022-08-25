using Axon.Common.LinearAlgebra;

namespace Axon.Optimization.Interfaces;

public interface IOptimizationContext
{
     public void AllocateMemoryForTrainingSet(MatrixStorage[] parameters, int samples);
}