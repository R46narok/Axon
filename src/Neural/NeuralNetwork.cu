#include "Neural/NeuralNetwork.cuh"
#include "Core/Random.cuh"
#include "Core/Activation.cuh"

#include <iostream>
#include <thrust/host_vector.h>

namespace Axon
{
    NeuralNetwork::NeuralNetwork(const NeuralNetworkDescriptor &descriptor)
        : m_Regularization(descriptor.Regularization), m_Layers(descriptor.Layers),
        m_Weights(),
        m_WeightsTransposed(),
        m_WeightsDerivatives()
    {
        InitializeMatrices(descriptor.DistributionBound);
        InitializeBiasTerms();
    }


    void NeuralNetwork::InitializeMatrices(float distribution)
    {
        uint32_t length = m_Layers.size() - 1;

        for (uint32_t i = 0; i < length; ++i)
        {
            m_Weights.emplace_back(m_Layers[i + 1], m_Layers[i] + 1);
            m_WeightsTransposed.emplace_back(m_Layers[i] + 1, m_Layers[i + 1]);
            m_WeightsDerivatives.emplace_back(m_Layers[i + 1], m_Layers[i] + 1);

            thrust::transform(
                    thrust::make_counting_iterator(0),
                    thrust::make_counting_iterator(m_Layers[i + 1] * (m_Layers[i] + 1)),
                    m_Weights[i].Begin(),
                    UniformDistribution<float>(-1.0f * distribution, distribution)
                    );

            m_WeightsTransposed[i] = m_Weights[i].Transpose();
        }
    }

    void NeuralNetwork::InitializeBiasTerms()
    {
        uint32_t l = m_Layers.size() - 1;
        for (uint32_t i = 0; i < l; ++i)
            m_Layers[i]++;
    }

    Matrix* NeuralNetwork::Feedforward(const Matrix* input, NeuralNetwork::FeedforwardDescriptor* descriptor)
    {
        uint32_t length = m_Layers.size() - 1;

        auto& preactivation = descriptor->preactivation;
        auto& preactivationDerivatives = descriptor->preactivationDerivatives;
        auto& activation = descriptor->activation;

        auto* pLast = (Matrix*)input;
        for (uint32_t i = 0; i < length; ++i)
        {
            preactivation[i] = pLast->Dot(m_WeightsTransposed[i]);
            Activation::Sigmoid(preactivation[i], preactivation[i]);

            if (descriptor->computeGradients && IsHiddenLayer(i + 1))
            {
                preactivationDerivatives[i] = pLast->Dot(m_WeightsTransposed[i]);
                Activation::Sigmoid(preactivationDerivatives[i], preactivationDerivatives[i], true);
            }

            if (IsNotOutputLayer(i + 1)) // no bias term added for the output layer
            {
                activation[i] = preactivation[i].InsertColumn(Matrix::Bias);
                pLast = &activation[i];
            }
            else
            {
                pLast = &preactivation[i];
            }
        }

        return pLast;
    }

    void NeuralNetwork::Backpropagate(const Matrix *input, const Matrix *output,
                                      NeuralNetwork::BackpropagationDescriptor *descriptor, NeuralNetwork::FeedforwardDescriptor* feedforwardDescriptor)
    {
        auto& errors = descriptor->errors;
        auto& errorsBiased = descriptor->errorsBiased;
        auto& errorsTransposed = descriptor->errorsTransposed;

        feedforwardDescriptor->computeGradients = true;
        auto* prediction = Feedforward(input, feedforwardDescriptor);

        auto &preactivation = feedforwardDescriptor->preactivation;
        auto &preactivationDerivatives = feedforwardDescriptor->preactivationDerivatives;
        auto &preactivationDerivativesBiased = feedforwardDescriptor->preactivationDerivativesBiased;

        errors[1] = *prediction - *output;
        for (uint32_t i = GetOutputLayerIdx() - 1; i >= GetInputLayerIdx() + 1; --i)
        {
            errorsBiased[i - 1] = errors[i].Dot(m_Weights[i]);
            preactivationDerivativesBiased[i - 1] = preactivationDerivatives[i - 1].InsertColumn(Matrix::Bias);

            errorsBiased[i - 1] = errorsBiased[i - 1] * preactivationDerivativesBiased[i - 1];
            errors[i - 1] = errorsBiased[i - 1].RemoveColumn(0);
        }

        auto* pLast = (Matrix*)input;
        auto samples = (float)input->GetRows();
        for (uint32_t i = 0; i < errors.size(); ++i)
        {
            errorsTransposed[i] = errors[i].Transpose();
            m_WeightsDerivatives[i] = errorsTransposed[i].Dot(*pLast);
            m_WeightsDerivatives[i] = m_WeightsDerivatives[i] * (1.0f / samples);

            pLast = &preactivation[i];
        }
    }

    void NeuralNetwork::AllocateFeedforwardDescriptor(NeuralNetwork::FeedforwardDescriptor* descriptor, uint32_t batchSize)
    {
        uint32_t layers = m_Weights.size();

        for (uint32_t i = 0; i < layers; ++i)
        {
            descriptor->preactivation.emplace_back(batchSize, m_Weights[i].GetRows());
            descriptor->preactivationDerivatives.emplace_back(batchSize, m_Weights[i].GetRows());
            descriptor->preactivationDerivativesBiased.emplace_back(batchSize, m_Weights[i].GetRows() + 1);
            descriptor->activation.emplace_back(batchSize, m_Weights[i].GetRows() + 1);
        }
    }

    void NeuralNetwork::AllocateBackpropagationDescriptor(NeuralNetwork::BackpropagationDescriptor *descriptor,
                                                          uint32_t batchSize)
    {
        uint32_t layers = m_Weights.size();

        for (uint32_t i = 0; i < layers; ++i)
        {
            descriptor->errors.emplace_back(batchSize, m_Weights[i].GetRows());
            descriptor->errorsTransposed.emplace_back(m_Weights[i].GetRows(), batchSize);
            descriptor->errorsBiased.emplace_back(batchSize, m_Weights[i].GetRows() + 1);
        }
    }
}