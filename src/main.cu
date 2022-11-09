#include "Core/Matrix.cuh"
#include "Core/Activation.cuh"
#include "Neural/NeuralNetwork.cuh"
#include "Data/Dataset.cuh"
#include "Data/Optimization/GradientDescent.cuh"

#include <thrust/host_vector.h>
#include <iostream>

using namespace Axon;

int main()
{
    uint32_t samples = 60000;

    Matrix input(samples, 784);
    Matrix bias(samples, 785);
    Matrix labels(samples, 10);

    Dataset dataset("mnist_train.csv",{
        {"Labels", 1, samples, 1.0f},
        {"Images", 784, samples, 256.0f}
    });
    dataset.Copy("Images", input);
    dataset.Copy("Labels", labels, 10);
    bias = input.InsertColumn(Matrix::Bias);

    NeuralNetworkDescriptor descriptor {};
    descriptor.Layers = {784, 300, 10};
    descriptor.DistributionBound = sqrt(6);
    descriptor.Regularization = 0.25f;

    NeuralNetwork network(descriptor);

    GradientDescent gradient;

    gradient.Optimize(network, bias, labels);

    samples = 10000;

    Matrix testInput(samples, 784);
    Matrix testBias(samples, 785);
    Matrix testLabels(samples, 10);

    Dataset test("mnist_train.csv",{
            {"Labels", 1, samples, 1.0f},
            {"Images", 784, samples, 256.0f}
    });
    test.Copy("Images", testInput);
    test.Copy("Labels", testLabels, 10);
    testBias = testInput.InsertColumn(Matrix::Bias);

    NeuralNetwork::FeedforwardDescriptor feedforwardDescriptor;
    network.AllocateFeedforwardDescriptor(&feedforwardDescriptor, testInput.GetRows());
    feedforwardDescriptor.computeGradients = false;
    auto* prediction = network.Feedforward(&testBias, &feedforwardDescriptor);

    thrust::host_vector<float> host(prediction->GetRows() * prediction->GetColumns());
    thrust::copy(prediction->Begin(), prediction->End(), host.begin());

    for (int i = 0; i < 30; i++)
    {
        for (int j = 0; j < prediction->GetColumns(); ++j)
        {
            std::cout << host[j * prediction->GetRows() + i] << " ";
        }

        std::cout << "\n";
    }

    return 0;
}