using System;
using Axon.Common.Functions;
using Axon.Common.Interfaces;
using Axon.Common.LinearAlgebra;
using Axon.Optimization.Interfaces;

namespace Axon.Optimization.Algorithms;

public class GradientDescent<T, U> : IOptimizationProcedure<T, U>
    where T : IOptimizationContext, new()
    where U : IPredictionContext, new()
{
    private readonly int _iterations;
    private readonly float _learningRate;

    public GradientDescent(float learningRate, int iterations)
    {
        _learningRate = learningRate;
        _iterations = iterations;
    }

    public void Optimize(ICostFunction<T, U> function, MatrixStorage x, MatrixStorage y)
    {
        InitializeTempParameters(function.Parameters, out var tempParameters);
        InitializeDerivatives(function.Parameters, out var derivatives);
        InitializeOptimizationContext(function.Parameters, x, out var optimizationContext);
        InitializePredictionContext(function.Parameters, x, out var predictionContext);
        
        for (int i = 0; i < _iterations; ++i)
        {
            var parameters = function.Parameters;
            for (int j = 0; j < parameters.Length; ++j)
            {
                var w = parameters[j];
                tempParameters[j].Buffer.Reset();
        
                w.Subtract(derivatives[j], tempParameters[j], _learningRate);
            }
        
            (function.Parameters, tempParameters) = (tempParameters, function.Parameters);
        
            for (int j = 0; j < parameters.Length; ++j)
                parameters[j].Transpose(function.ParametersTransposed[j]);
            
            if (i == 0 ||(i + 1) % 500 == 0 || i + 1 == _iterations) 
                Console.WriteLine($"Iteration {i + 1}");
            derivatives = function.ComputeDerivatives(x, y, optimizationContext, predictionContext);
        }
    }

    private void InitializeOptimizationContext(MatrixStorage[] parameters, MatrixStorage x, out T context)
    {
        context = new T();
        context.AllocateMemoryForTrainingSet(parameters, x.Rows);
    }
    
    private void InitializePredictionContext(MatrixStorage[] parameters, MatrixStorage x, out U context)
    {
        context = new U();
        context.AllocateMemoryForPredictionBatch(parameters, x.Rows);
    }
    
    private void InitializeTempParameters(MatrixStorage[] parameters, out MatrixStorage[] tempParameters)
    {
        var length = parameters.Length;
        
        tempParameters = new MatrixStorage[length];
        for (int i = 0; i < length; ++i)
        {
            tempParameters[i] = new MatrixStorage(parameters[i].Rows, parameters[i].Columns);
        }
    }

    private void InitializeDerivatives(MatrixStorage[] parameters, out MatrixStorage[] derivatives)
    {
        var length = parameters.Length;
                
        derivatives = new MatrixStorage[length];
        for (int i = 0; i < length; ++i)
        {
            derivatives[i] = new MatrixStorage(parameters[i].Rows, parameters[i].Columns);
        }
    }
}