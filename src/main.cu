#include "Core/Matrix.cuh"
#include "Core/Activation.cuh"
#include "Neural/NeuralNetwork.cuh"

#include <thrust/host_vector.h>
#include <iostream>

using namespace Axon;

int main()
{
    NeuralNetworkDescriptor descriptor {};

    descriptor.Layers = {700, 300, 10};
    NeuralNetwork network(descriptor);
    return 0;
}