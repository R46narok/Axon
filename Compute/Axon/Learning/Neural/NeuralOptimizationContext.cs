using Axon.Common.Functions;
using Axon.Optimization;
using Axon.Common.Buffers;
using Axon.Common.LinearAlgebra;
using Axon.Helpers;
using Axon.Optimization.Interfaces;

// ReSharper disable NotAccessedField.Local

namespace Axon.Learning.Neural;

public class NeuralOptimizationContext : IOptimizationContext
{
   private BufferBatch _errorBatch;
   private BufferBatch _errorsTransposedBatch;
   private BufferBatch _errorsBiasedBatch;
   
   private MatrixStorage[] _errors;
   private MatrixStorage[] _errorsTransposed;
   private MatrixStorage[] _errorsBiased;
   
   public MatrixStorage[] Errors => _errors;
   public MatrixStorage[] ErrorsTransposed => _errorsTransposed;
   public MatrixStorage[] ErrorsBiased => _errorsBiased;

   public void AllocateMemoryForTrainingSet(MatrixStorage[] weights, int samples)
   {
       int layers = weights.Length + 1;
       
       BatchMatrixHelper.InitializeBatchAsMatrixArray(out _errorBatch, out _errors,
           layers - 1, i => samples, i => weights[i].Rows, 
           BufferDataType.Double, "errors");
       
       BatchMatrixHelper.InitializeBatchAsMatrixArray(out _errorsTransposedBatch, out _errorsTransposed, 
           layers - 1, i => weights[i].Rows, i => samples, 
           BufferDataType.Double, "errorsTransposed");
       
       BatchMatrixHelper.InitializeBatchAsMatrixArray(out _errorsBiasedBatch, out _errorsBiased, 
           layers - 2, i => samples, i=> weights[i].Rows + 1, 
           BufferDataType.Double, "errorsTransposed");
   }
}