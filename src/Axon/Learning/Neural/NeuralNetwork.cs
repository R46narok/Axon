using System;
using Axon.Common.Buffers;
using Axon.Optimization.Interfaces;
using Axon.Common.Functions;
using Axon.Common.LinearAlgebra;
using Axon.Exceptions;

namespace Axon.Learning.Neural;

public class NeuralNetwork : ICostFunction<NeuralOptimizationContext, NeuralPredictionContext>
{
    private readonly int[] _layers;
    private readonly float _regularization;
    private readonly IMatrixHardwareAcceleration _acceleration;
    private readonly IBufferAllocator _allocator;
    
    private MatrixStorage[] _weights = null!;
    private MatrixStorage[] _weightsReg = null!;
    private MatrixStorage[] _weightsTransposed = null!;
    private MatrixStorage[] _derivatives = null!;
    
    public MatrixStorage[] Parameters { get => _weights; set => _weights = value; } // TODO: Add set validation
    public MatrixStorage[] ParametersTransposed { get => _weightsTransposed; set => _weightsTransposed = value; } // TODO: Add set validation
    
    public NeuralNetwork(NeuralNetworkOptions options)
    {
        if (options is null) throw new ArgumentNullException(nameof(options));

        _layers = options.Layers;
        _regularization = options.Regularization;
        _acceleration = options.Acceleration;
        _allocator = options.Allocator;
        
        OutputLayerIdx = _layers.Length - 1;
        
        ValidateNetworkArchitecture();
        InitializeWeights(options.Distribution);
        InitializeDerivatives();
        InitializeBiasNeurons();
    }

    private void ValidateNetworkArchitecture()
    {
        if (_layers.Length < 2) throw new ArchitectureException();
    }

    private void InitializeWeights(UniformDistribution distribution)
    {
        int length = _layers.Length - 1;
        _weights = new MatrixStorage[length];
        _weightsReg = new MatrixStorage[length];
        _weightsTransposed = new MatrixStorage[length];

        var compute = MatrixComputeContext.Create(_acceleration);
        for (int i = 0; i < length; ++i)
        {
            var cpuBuffer = new float[_layers[i + 1] * (_layers[i] + 1)];
            for (int j = 0; j < cpuBuffer.Length; ++j)
                cpuBuffer[j] = distribution.Next();

            _weights[i] = new MatrixStorage(_layers[i + 1], _layers[i] + 1, _allocator);
            _weightsTransposed[i] = new MatrixStorage(_layers[i] + 1, _layers[i + 1], _allocator);
            
            _weightsReg[i] = new MatrixStorage(_layers[i + 1], _layers[i] + 1, _allocator);
            
            _weights[i].Buffer.Upload(cpuBuffer);
            compute.PerformOn(_weights[i]).Into(_weightsTransposed[i]).Transpose();
        }
    }

    private void InitializeDerivatives()
    {
        int length = _layers.Length - 1;
        _derivatives = new MatrixStorage[length];

        for (int i = 0; i < length; ++i)
            _derivatives[i] = new MatrixStorage(_layers[i + 1], _layers[i] + 1, _allocator);
    }
    
    /// <summary>
    /// Adds bias neuron to each layer except the last.
    /// </summary>
    private void InitializeBiasNeurons()
    {
        int l = _layers.Length - 1;
        for (int i = 0; i < l; ++i)
        {
            _layers[i]++;
        }
    }

    private const int InputLayerIdx = 0;
    private readonly int OutputLayerIdx;

    private bool IsOutputLayer(int idx) => idx == OutputLayerIdx;
    private bool IsInputLayer(int idx) => idx == InputLayerIdx;
    private bool IsHiddenLayer(int idx) => !IsInputLayer(idx) && !IsOutputLayer(idx);
    
    private bool IsNotOutputLayer(int idx) => !IsOutputLayer(idx);
    private bool IsNotInputLayer(int idx) => !IsInputLayer(idx);
    private bool IsNotHiddenLayer(int idx) => !IsHiddenLayer(idx);

    
    /// <summary>
    /// Computes the output of a neural network based on a given input
    /// using the forward propagation algorithm.
    /// </summary>
    /// <param name="x">Matrix for the input layer with already added bias column</param>
    /// <param name="computeGradients"></param>
    /// <returns>[training samples x number of output units] matrix, which contains the predictions</returns>
    public MatrixStorage FeedForward(MatrixStorage x, NeuralPredictionContext predictionContext, bool computeGradients = false)
    {
        var compute = MatrixComputeContext.Create(_acceleration);
        compute.PushRange("Forward propagation");

        var preactivation = predictionContext.Preactivation;
        var preactivationGradient = predictionContext.PreactivationGradient;
        var activation = predictionContext.Activation;
        
        var last = x;
        var length = _layers.Length - 1; // excluding the first(input) layer
        for (int i = 0; i < length; ++i)
        {
            compute.PerformOn(last).And(_weightsTransposed[i]).MultiplyInto(preactivation[i]); // preactivation of the current layer
            compute.PerformOnSelf(preactivation[i]).ApplySigmoidFunction(); // activation of the current layer

            if (computeGradients && IsHiddenLayer(i + 1))
            {
                compute.PerformOn(last).And(_weightsTransposed[i]).MultiplyInto(preactivationGradient[i]);
                compute.PerformOnSelf(preactivationGradient[i]).ApplySigmoidGradientFunction();
            }
            
            if (IsNotOutputLayer(i + 1)) // no bias term added for the output layer
            {
                compute.PerformOn(preactivation[i]).Into(activation[i]).InsertColumn(1.0f); // fully activated layer (added bias)
                last = activation[i];
            }
            else
            {
                last = preactivation[i];
            }
        }

        compute.PopRange();
        return last; // Output layer predictions
    }

    /// <summary>
    /// Vectorized impl of backpropagation 
    /// </summary>
    /// <param name="x"></param>
    /// <param name="y"></param>
    /// <param name="optimizationContext"></param>
    /// <param name="predictionContext"></param>
    private void Backpropagation(MatrixStorage x, MatrixStorage y, NeuralOptimizationContext optimizationContext, NeuralPredictionContext predictionContext)
    {
        ResetErrorTerms();
        
        var compute = MatrixComputeContext.Create(_acceleration);
        var prediction = FeedForward(x, predictionContext, true);

        var errors = optimizationContext.Errors;
        var errorsBiased = optimizationContext.ErrorsBiased;
        var errorsTransposed = optimizationContext.ErrorsTransposed;

        var preactivation = predictionContext.Preactivation;
        var preactivationGradient = predictionContext.PreactivationGradient;
        var preactivationGradientBiased = predictionContext.PreactivationGradientBiased;
        
        compute.PerformOn(prediction).And(y).PointwiseSubtractInto(errors[1]);
        for (int i = OutputLayerIdx - 1; i >= InputLayerIdx + 1; --i)
        {
            compute.PerformOn(errors[i]).And(_weights[i]).MultiplyInto(errorsBiased[i - 1]);
            compute.PerformOn(preactivationGradient[i - 1]).Into(preactivationGradientBiased[i - 1]).InsertColumn(1.0f);

            compute.PerformOn(errorsBiased[i - 1]).And(preactivationGradientBiased[i - 1]).PointwiseMultiplyInto(errorsBiased[i - 1]);
            compute.PerformOn(errorsBiased[i - 1]).Into(errors[i - 1]).RemoveColumn();
        }

        var layer = x;
        int samples = x.Rows;
        for (int i = 0; i < errors.Length; ++i)
        {
            compute.PerformOn(errors[i]).Into(errorsTransposed[i]).Transpose();
            compute.PerformOn(errorsTransposed[i]).And(layer).MultiplyInto(_derivatives[i]);
            compute.PerformOnSelf(_derivatives[i]).MultiplyBy(1.0f / samples);
            
            layer = preactivation[i];
        }
        
        // Regularization
        for (int i = InputLayerIdx; i < OutputLayerIdx; ++i)
        {
            compute.PerformOn(_weights[i]).Into(_weightsReg[i]).MultiplyBy(_regularization / samples);
            compute.PerformOn(_derivatives[i]).And(_weightsReg[i]).AddInto(_derivatives[i]);
        }
    }

    public float ComputeCost(MatrixStorage x, MatrixStorage y, NeuralPredictionContext predictionContext)
    {
        // var h = FeedForward(x, predictionContext);
        // var compute = MatrixComputeContext.Create(_acceleration);
        //
        // const int hIdx = 0;
        // const int yIdx = 1;
        //
        // var invertedOutput = predictionContext.InvertedOutput;
        //
        // compute.PerformOn(y).Into(invertedOutput[yIdx]).MultiplyBy(-1.0f);
        // compute.PerformOn(h).Into(invertedOutput[hIdx]).MultiplyBy(-1.0f);
        // compute.PerformOnSelf(h).Log();
        //
        // invertedOutput[yIdx].PointwiseMultiply(h, h);
        // invertedOutput[yIdx].Add(1.0f, invertedOutput[yIdx]);
        // invertedOutput[hIdx].Add(1.0f, invertedOutput[hIdx]);
        //
        // invertedOutput[hIdx].PointwiseLog(invertedOutput[hIdx]);
        // invertedOutput[yIdx].PointwiseMultiply(invertedOutput[hIdx], invertedOutput[yIdx]);
        //
        // h.Subtract(invertedOutput[yIdx], h);
        //
        // int samples = x.Rows;
        // var cost = h.Sum();
        //
        // return cost / samples;
        return 0.0f;
    }


    public MatrixStorage[] ComputeDerivatives(MatrixStorage x, MatrixStorage y, 
        NeuralOptimizationContext optimizationContext, NeuralPredictionContext predictionContext)
    {
        Backpropagation(x, y, optimizationContext, predictionContext);
        return _derivatives;
    }

    private void ResetErrorTerms()
    {
        int length = _derivatives.Length;
        for (int i = 0; i < length; ++i)
            _derivatives[i].Buffer.Reset();
    }
}