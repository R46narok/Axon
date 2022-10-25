#include "Core/Matrix.cuh"
#include "Core/Activation.cuh"
#include "Data/Dataset.cuh"
#include "Neural/NeuralNetwork.cuh"

#include <thrust/host_vector.h>
#include <iostream>

using namespace Axon;

int main()
{
    uint32_t samples = 60000;

    Matrix input(samples, 784);
    Matrix labels(samples, 10);
    Dataset dataset("mnist_train.csv",{
        {"Labels", 1, samples, 1.0f},
        {"Images", 784, samples, 256.0f}
    });
    dataset.Copy("Images", input);
    dataset.Copy("Labels", labels, 10);

    NeuralNetworkDescriptor descriptor {};

    descriptor.Layers = {700, 300, 10};
    NeuralNetwork network(descriptor);
    return 0;
}