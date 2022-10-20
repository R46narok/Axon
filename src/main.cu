#include "Core/Matrix.cuh"
#include "Core/Activation.cuh"

#include <thrust/host_vector.h>
#include <iostream>

int main()
{
    Axon::Matrix a(3, 4);

    thrust::sequence(a.Begin(), a.End());

    Axon::Matrix out(3, 4);
    Axon::Activation::Sigmoid(out, a, true);

    thrust::host_vector<float> host(3 * 4);
    thrust::copy(out.Begin(), out.End(), host.begin());

    for (int i = 0; i < 3 * 4; ++i)
        std::cout << host[i] << std::endl;

    return 0;
}