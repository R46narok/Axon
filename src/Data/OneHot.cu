#include "Core/Matrix.cuh"
#include "Data/OneHot.cuh"

#include <thrust/host_vector.h>

namespace Axon
{
    void OneHot::Encode(thrust::host_vector<float>& input, thrust::host_vector<float>& output, int classes)
    {
        for (int i = 0; i < input.size(); ++i)
        {
            output[i * classes + (int)input[i]] = 1.0f;
        }
    }
}