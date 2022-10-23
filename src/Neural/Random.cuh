#ifndef _AXON_UNIFORM_DISTRIBUTION_H
#define _AXON_UNIFORM_DISTRIBUTION_H

#include "Core/Interop.cuh"

#include <thrust/random.h>
#include <cmath>

namespace Axon
{
    template<class T>
    struct UniformDistribution
    {
        float low, high;

        UniformDistribution(float l, float h)
        {
            low = l;
            high = h;
        }

        __device__ T operator()(int idx)
        {
            thrust::default_random_engine randEng;
            thrust::uniform_real_distribution<T> dist(low, high);
            randEng.discard(idx);
            return dist(randEng);
        }
    };
}

#endif //_AXON_UNIFORM_DISTRIBUTION_H
