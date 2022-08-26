using System;
using System.IO;
using System.Net.Http;
using System.Threading.Tasks;
using Axon.Common.Buffers;
using Axon.Common.LinearAlgebra;
using Axon.Cuda.Common.Buffers;
using Axon.Cuda.Operations;
using Axon.Helpers.FileParsers.Csv;
using Axon.Learning.Neural;
using Axon.Optimization.Algorithms;
using Xunit;

namespace Axon.Tests.NeuralNetwork;

public class NeuralNetworkMnist
{
    private const float Regularization = 6.75f;
    private const int TrainingSamples = 60000;
    private const int TestSamples = 10000;

    private readonly IMatrixComputationSelectionStage _computation;
    private readonly IMatrixHardwareAcceleration _acceleration;
    private readonly IBufferAllocator _allocator;
    
    public NeuralNetworkMnist()
    {
        _acceleration = new CudaMatrixAcceleration();
        _allocator = new GlobalMemoryAllocator();
        _computation = MatrixComputeContext.Create(_acceleration);
    }

    private const int InputLayer = 784;
    private const int HiddenLayer = 512;
    private const int OutputLayer = 10;
    
    private Learning.Neural.NeuralNetwork CreateMnistNetwork()
    {
        return NeuralNetworkBuilder
            .Create()
            .AddLayer(InputLayer)
            .AddLayer(HiddenLayer)
            .AddLayer(OutputLayer)
            .UseDistribution((float) Math.Sqrt(6))
            .UseRegularization(Regularization)
            .WithBufferAllocator(_allocator)
            .WithHardwareAcceleration(_acceleration)
            .Build();
    }
    
    [Theory]
    [InlineData(2.9f, 2000)]
    public async Task GradientDescent(float learningRate, int iterations)
    {
        var network = CreateMnistNetwork();

        var (x, y) = await LoadDataSet();
        
        var procedure = new GradientDescent<NeuralOptimizationContext, NeuralPredictionContext>(learningRate, iterations, _computation);
        procedure.Optimize(network, x, y);

        (x, y) = await LoadTestSet();
        var predictionContext = new NeuralPredictionContext();
        predictionContext.AllocateMemoryForPredictionBatch(network.Parameters, TestSamples);

        var prediction = network.FeedForward(x, predictionContext);
        
        var cpuLabels = y.Buffer.CopyToHost();
        var cpu = prediction.Buffer.CopyToHost();

        int correct = 0;
        for (int i = 0; i < cpuLabels.Length; ++i)
        {
            var idx = (int) cpuLabels[i] * TestSamples + i;
            if (Math.Round(cpu[idx]) == 1.0f) 
                correct++;
        }

        var accuracy = (float) correct / TestSamples * 100;
        Assert.InRange(accuracy, 90.0f, 100.0f);
    }

    private async Task<(MatrixStorage, MatrixStorage)> LoadDataSet()
    {
        var filename = "mnist_train.csv";
        DownloadIfDoesNotExist(filename, "https://github.com/R46narok/DataSets/raw/main/Mnist/mnist_train.csv").Wait();
        
        var parser = new CsvParser(_allocator);
        var result = await parser.ParseAsync(filename, options =>
        {
            options.AddLabelMatrixDefinition("labels", TrainingSamples, OutputLayer);
            options.AddMatrixDefinition("data", TrainingSamples, InputLayer, 256.0f);
        });

        var x = result["data"];
        var y = result["labels"];
        
        var xBiased = new MatrixStorage(TrainingSamples, InputLayer + 1, _allocator);
        _computation.PerformOn(x).Into(xBiased).InsertColumn(1.0f);

        return (xBiased, y);
    }

    private async Task<(MatrixStorage, MatrixStorage)> LoadTestSet()
    {
        var filename = "mnist_test.csv";
        DownloadIfDoesNotExist(filename, "https://github.com/R46narok/DataSets/raw/main/Mnist/mnist_test.csv").Wait();
        
        var parser = new CsvParser(_allocator);
        var result = await parser.ParseAsync(filename, options =>
        {
            options.AddMatrixDefinition("labels", TestSamples, 1);
            options.AddMatrixDefinition("data", TestSamples, InputLayer, 256.0f);
        });
       
        var x = result["data"];
        var y = result["labels"];
        
        var xBiased = new MatrixStorage(TestSamples, InputLayer + 1, _allocator);
        _computation.PerformOn(x).Into(xBiased).InsertColumn(1.0f);
       
        return (xBiased, y);
    }
    
    private async Task DownloadIfDoesNotExist(string file, string url)
    {
        if (!File.Exists(file))
        {
            var uri = new Uri(url);
            HttpClient client = new HttpClient();
            var response = await client.GetAsync(uri);
            await using var fs = new FileStream(file, FileMode.CreateNew);
            await response.Content.CopyToAsync(fs);
        }
    }
}