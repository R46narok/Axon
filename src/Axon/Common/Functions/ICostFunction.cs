using Axon.Common.Interfaces;
using Axon.Common.LinearAlgebra;
using Axon.Optimization.Interfaces;

namespace Axon.Common.Functions;

public interface ICostFunction<T, U> 
   where T : IOptimizationContext, new()
   where U : IPredictionContext, new()
{
   public MatrixStorage[] Parameters { get; set; }
   public MatrixStorage[] ParametersTransposed { get; set; }
   
   public float ComputeCost(MatrixStorage x, MatrixStorage y, U predictionContext);
   public MatrixStorage[] ComputeDerivatives(MatrixStorage x, MatrixStorage y, T optimizationContext, U predictionContext);
}