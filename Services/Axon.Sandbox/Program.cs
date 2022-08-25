using System.Diagnostics;
using System.Threading.Channels;
using Axon.Common.LinearAlgebra;
using Axon.Cuda.Buffers;
using Axon.Cuda.Common.Interop;
using Axon.Cuda.Operations;
using Axon.Learning.Neural;
using Axon.Optimization.Algorithms;
using Microsoft.VisualBasic.FileIO;

MatrixStorage.BufferFactory = new GlobalMemoryAllocator();
MatrixStorage.Operations = new GpuMatrixOperations();

var nn = NeuralNetworkBuilder.Create()
    .AddLayer(784)
    .AddLayer(512)
    .AddLayer(10)
    .UseDistribution((float) Math.Sqrt(6))
    .UseRegularization(6.75f)
    .Build();

int samples = 60000;

var x = new MatrixStorage(samples, 784);
var y = new MatrixStorage(samples, 10);

using var rd = new StreamReader("mnist_train.csv");
int line = -1;

var cpuX = new float[samples * 784];
var cpuY = new float[samples * 10];

while (!rd.EndOfStream && line <= samples)
{
    line++;
    var splits = rd!.ReadLine().Split(',');
    if (line > 0 && line <= samples)
    {
        var dd = Array.ConvertAll(splits, float.Parse);
    
        var label = dd[0];
        cpuY[(int) label * samples + (line - 1)] = 1;
        // cpuY[(line - 1) * 10 + (int) label] = 1;
        for (int i = 0; i < 784; ++i)
        {
            cpuX[i * samples + (line - 1)] = dd[1 + i] / 256.0f;
            // cpuX[(line - 1) * 784 + i] = dd[1 + i] / 256.0f;
        }
    }
}

x.Buffer.Upload(cpuX);
y.Buffer.Upload(cpuY);

var assert = new CudaAssert();

x = x.InsertColumn(1.0f);
var procedure = new GradientDescent<NeuralOptimizationContext, NeuralPredictionContext>(2.9f, 2000);
procedure.Optimize(nn, x, y);

using var rd2 = new StreamReader("mnist_test.csv");
line = -1;
int correct = 0;
int wrong = 0;

var predictionContext = new NeuralPredictionContext();
predictionContext.AllocateMemoryForPredictionBatch(nn.Parameters, 1);
MatrixStorage prediction = new MatrixStorage(1, 784);
MatrixStorage predictionBiased = new MatrixStorage(1, 784 + 1);

while (!rd2.EndOfStream)
{
    line++;
    var splits = rd2.ReadLine()!.Split(',');
    if (line > 0)
    {
        var dd = Array.ConvertAll(splits, float.Parse);
        var cpuData = new float[784];
        var label = dd[0];
        for (int i = 0; i < 784; ++i)
            cpuData[i] = dd[1 + i] / 256.0f;
        prediction.Buffer.Upload(cpuData);
        prediction.InsertColumn(1.0f, predictionBiased);
        
        var result = nn.FeedForward(predictionBiased, predictionContext);
        var buffer = result.Buffer.Read()!;
        var output = Math.Round(buffer[(int)label]);
        
        if (output == 0) wrong++;
        else correct++;
    }
}

Console.WriteLine("Accuracy: {0:F2}", (float)correct / (correct + wrong) * 100);
Console.WriteLine($"Correct {correct} from {correct + wrong}");