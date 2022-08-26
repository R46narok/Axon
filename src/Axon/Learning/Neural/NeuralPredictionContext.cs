using Axon.Optimization.Interfaces;
using Axon.Common.Buffers;
using Axon.Common.Interfaces;
using Axon.Common.LinearAlgebra;
using Axon.Helpers;

namespace Axon.Learning.Neural;

public class NeuralPredictionContext : IPredictionContext
{
    private BufferBatch _preactivationBatch;
    private BufferBatch _preactivationGradientBatch;
    private BufferBatch _preactivationGradientBiasedBatch;
    private BufferBatch _activationBatch;
    private BufferBatch _invertedOutputBatch;
    
    private MatrixStorage[] _preactivation;
    private MatrixStorage[] _preactivationGradient;
    private MatrixStorage[] _preactivationGradientBiased;
    private MatrixStorage[] _activation;
    private MatrixStorage[] _invertedOutput;
    
    public MatrixStorage[] Preactivation => _preactivation;
    public MatrixStorage[] Activation => _activation;
    public MatrixStorage[] PreactivationGradient => _preactivationGradient;
    public MatrixStorage[] PreactivationGradientBiased => _preactivationGradientBiased;
    public MatrixStorage[] InvertedOutput => _invertedOutput;
    
    public void AllocateMemoryForPredictionBatch(MatrixStorage[] weights, int batchSize)
    {
       int layers = weights.Length + 1;

       var allocator = weights[0].Allocator;
       
        BatchMatrixHelper.InitializeBatchAsMatrixArray(allocator, out _preactivationBatch, out _preactivation, 
            layers - 1, i => batchSize, i => weights[i].Rows, 
            BufferDataType.Double, "preactivation");
               
        BatchMatrixHelper.InitializeBatchAsMatrixArray(allocator, out _preactivationGradientBatch, out _preactivationGradient, 
            layers - 1, i => batchSize, i => weights[i].Rows, 
            BufferDataType.Double, "preactivation");
        
        BatchMatrixHelper.InitializeBatchAsMatrixArray(allocator, out _preactivationGradientBiasedBatch, out _preactivationGradientBiased, 
            layers - 1, i => batchSize, i => weights[i].Rows + 1, 
            BufferDataType.Double, "preactivation");
        
        BatchMatrixHelper.InitializeBatchAsMatrixArray(allocator, out _activationBatch, out _activation, 
            layers - 1, i => batchSize, i=> weights[i].Rows + 1, 
            BufferDataType.Double, "activation");
        
        BatchMatrixHelper.InitializeBatchAsMatrixArray(allocator, out _invertedOutputBatch, out _invertedOutput, 
              2, i => batchSize, i=> weights[1].Rows, 
              BufferDataType.Double, "errorsTransposed");
    }
}