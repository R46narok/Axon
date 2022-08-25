using Axon.Common.Buffers;
using Axon.Common.LinearAlgebra;

namespace Axon.Learning.Neural;

public class NeuralNetworkOptions
{
    public float Regularization { get; set; }
    public int[] Layers { get; set; }
    public UniformDistribution Distribution { get; set; }

    public IMatrixHardwareAcceleration Acceleration { get; set; }
    public IBufferAllocator Allocator { get; set; }
    
    public NeuralNetworkOptions(int[] layers, UniformDistribution distribution, float regularization, IMatrixHardwareAcceleration acceleration, IBufferAllocator allocator)
    {
        Layers = layers;
        Distribution = distribution;
        Regularization = regularization;
        Acceleration = acceleration;
        Allocator = allocator;
    }
}