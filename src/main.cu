#include "Core/Matrix.cuh"
#include "Core/Activation.cuh"
#include "Neural/NeuralNetwork.cuh"

#include <thrust/host_vector.h>
#include <iostream>

int main()
{
    Axon::NeuralNetworkDescriptor descriptor {};
    descriptor.Layers = {700, 300, 10};
    Axon::NeuralNetwork network(descriptor);
    return 0;
}