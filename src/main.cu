#include "Core/Matrix.cuh"

#include <thrust/host_vector.h>
#include <iostream>

int main()
{
    Axon::Matrix a(3, 3);

    thrust::sequence(a.Begin(), a.End());

    Axon::Matrix out(3, 3);
    out = a.Transpose();

    thrust::host_vector<float> host(3 * 3);
    thrust::copy(out.Begin(), out.End(), host.begin());

    for (int i = 0; i < 3 * 3; ++i)
        std::cout << host[i] << std::endl;

    return 0;
}