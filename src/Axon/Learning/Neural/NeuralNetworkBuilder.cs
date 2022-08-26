using Axon.Common.Buffers;
using Axon.Common.LinearAlgebra;

namespace Axon.Learning.Neural;

public class NeuralNetworkBuilder
{
    private List<int> _layers = new();
    private float _regularization = 0.0f;
    private float _distribution = 0.0f;
    private IBufferAllocator _allocator;
    private IMatrixHardwareAcceleration _acceleration;
    
    private NeuralNetworkBuilder()
    {
        
    }

    public static NeuralNetworkBuilder Create()
    {
        return new NeuralNetworkBuilder();
    }

    public NeuralNetworkBuilder AddLayer(int neurons)
    {
        _layers.Add(neurons);
        return this;
    }

    public NeuralNetworkBuilder UseDistribution(float distribution)
    {
        _distribution = distribution;
        return this;
    }

    public NeuralNetworkBuilder UseRegularization(float regularization)
    {
        _regularization = regularization;
        return this;
    }

    public NeuralNetworkBuilder WithBufferAllocator(IBufferAllocator allocator)
    {
        _allocator = allocator;
        return this;
    }

    public NeuralNetworkBuilder WithHardwareAcceleration(IMatrixHardwareAcceleration acceleration)
    {
        _acceleration = acceleration;
        return this;
    }
    
    public NeuralNetwork Build()
    {
        var options =
            new NeuralNetworkOptions(_layers.ToArray(), new UniformDistribution(_distribution), _regularization, _acceleration, _allocator);
        return new NeuralNetwork(options);
    }
}