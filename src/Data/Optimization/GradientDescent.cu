#include "Core/Matrix.cuh"
#include "Neural/NeuralNetwork.cuh"
#include "Data/Optimization/GradientDescent.cuh"

#include <algorithm>
#include <iostream>

namespace Axon
{
    GradientDescent::GradientDescent(uint32_t iterations, float learningRate)
        : m_Iterations(iterations), m_LearningRate(learningRate)
    {
    }

    void GradientDescent::Optimize(NeuralNetwork &network, const Matrix &input, const Matrix &output) const
    {
        NeuralNetwork::FeedforwardDescriptor feedforwardDescriptor;
        NeuralNetwork::BackpropagationDescriptor backpropagationDescriptor;

        network.AllocateFeedforwardDescriptor(&feedforwardDescriptor, input.GetRows());
        network.AllocateBackpropagationDescriptor(&backpropagationDescriptor, input.GetRows());

        std::vector<Matrix> tempWeights;

        auto& weights = network.GetWeights();
        for (auto & weight : weights)
            tempWeights.emplace_back(weight.GetRows(), weight.GetColumns());

        for (uint32_t i = 0; i < m_Iterations; ++i)
        {
            if ((i + 1) % 100 == 0)
                std::cout << "Iteration " << i + 1 << std::endl;

            weights = network.GetWeights();
            auto& weightsTransposed = network.GetWeightsTransposed();
            auto& derivatives = network.GetWeightsDerivatives();

            for (uint32_t j = 0; j < weights.size(); ++j)
            {
                derivatives[j] = derivatives[j] * m_LearningRate;
                tempWeights[j] = weights[j] - derivatives[j];
            }

            weights.swap(tempWeights);

            for (uint32_t j = 0; j < weights.size(); ++j)
                weightsTransposed[j] = weights[j].Transpose();

            network.Backpropagate(&input, &output, &backpropagationDescriptor, &feedforwardDescriptor);
        }

    }
}

