#ifndef _AXON_ONE_HOT_H
#define _AXON_ONE_HOT_H

#include "Core/Library.cuh"

#include <thrust/host_vector.h>
namespace Axon
{
    class AXON_API Matrix;

    class AXON_API OneHot
    {
    public:
        static void Encode(thrust::host_vector<float>& input, thrust::host_vector<float>& output, int classes);
    };
}

#endif //_AXON_ONE_HOT_H
