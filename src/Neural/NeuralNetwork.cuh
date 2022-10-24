//
// Created by Acer on 19.10.2022 г..
//
#ifndef _AXON_NEURAL_NETWORK_H
#define _AXON_NEURAL_NETWORK_H

#include "Core/Interop.cuh"
#include "Core/Matrix.cuh"

#include <vector>

namespace Axon
{
    struct NeuralNetworkDescriptor
    {
        float Regularization;
        float DistributionBound;
        std::vector<int> Layers;
    };

    // TODO: Rename to DenseLayers or sth like that
    class AXON_API NeuralNetwork
    {
    public:
        explicit NeuralNetwork(const NeuralNetworkDescriptor& descriptor);
        ~NeuralNetwork() noexcept = default;

        struct FeedforwardDescriptor
        {
            bool computeGradients;
            std::vector<Matrix> preactivation;
            std::vector<Matrix> preactivationDerivatives;
            std::vector<Matrix> preactivationDerivativesBiased;
            std::vector<Matrix> activation;
        };

        struct BackpropagationDescriptor
        {
            std::vector<Matrix> errors;
            std::vector<Matrix> errorsTransposed;
            std::vector<Matrix> errorsBiased;
        };

        [[nodiscard]] Matrix& Feedforward(const Matrix& input, FeedforwardDescriptor& descriptor);
        void Backpropagate(const Matrix& input, const Matrix& output,
                           BackpropagationDescriptor& descriptor, FeedforwardDescriptor& feedforwardDescriptor);

        [[nodiscard]] inline const std::vector<Matrix>& GetWeights() const noexcept { return m_Weights; }

        [[nodiscard]] inline bool IsOutputLayer(int idx) const noexcept { return idx == GetOutputLayerIdx(); }
        [[nodiscard]] inline bool IsInputLayer(int idx) const noexcept { return idx == GetInputLayerIdx(); }
        [[nodiscard]] inline bool IsHiddenLayer(int idx) const noexcept { return !IsInputLayer(idx) && !IsOutputLayer(idx); }

        [[nodiscard]] inline bool IsNotOutputLayer(int idx) const noexcept { return !IsOutputLayer(idx); }
        [[nodiscard]] inline bool IsNotInputLayer(int idx) const noexcept { return !IsInputLayer(idx); }
        [[nodiscard]] inline bool IsNotHiddenLayer(int idx) const noexcept { return !IsHiddenLayer(idx); }

        [[nodiscard]] inline constexpr uint32_t GetInputLayerIdx() const noexcept { return 0; }
        [[nodiscard]] inline uint32_t GetOutputLayerIdx() const noexcept { return m_Layers.size() - 1; }
    private:
        void InitializeMatrices(float distribution);
        void InitializeBiasTerms();
    private:
        std::vector<int> m_Layers;
        std::vector<Matrix> m_Weights;
        std::vector<Matrix> m_WeightsTransposed; // TODO: Maybe delete this
        std::vector<Matrix> m_WeightsDerivatives;

        float m_Regularization;
    };
}

#endif //_AXON_NEURAL_NETWORK_H
