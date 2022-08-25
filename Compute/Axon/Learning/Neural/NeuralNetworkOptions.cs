namespace Axon.Learning.Neural;

public class NeuralNetworkOptions
{
    public float Regularization { get; set; }
    public int[] Layers { get; set; }
    public UniformDistribution Distribution { get; set; }

    public NeuralNetworkOptions(int[] layers, UniformDistribution distribution, float regularization)
    {
        Layers = layers;
        Distribution = distribution;
        Regularization = regularization;
    }
}