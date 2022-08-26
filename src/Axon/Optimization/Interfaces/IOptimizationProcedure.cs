using Axon.Common.Functions;
using Axon.Common.Interfaces;
using Axon.Common.LinearAlgebra;

namespace Axon.Optimization.Interfaces;

public interface IOptimizationProcedure<T, U>
    where T : IOptimizationContext, new()
    where U : IPredictionContext, new()
{
    public void Optimize(ICostFunction<T, U> function, MatrixStorage x, MatrixStorage y);
}