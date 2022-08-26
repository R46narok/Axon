using System;

namespace Axon.Learning.Neural;

public class UniformDistribution
{
    private readonly float _bound;

    public UniformDistribution(float bound)
    {
        _bound = bound;
    }

    public float Next()
    {
        var axis = (float)(Random.Shared.Next(0, 2) * 2 - 1);
        var distribution = Random.Shared.NextSingle();
        var value = axis * _bound * distribution;

        return value;
    }
}