//
// Created by Acer on 19.10.2022 г..
//
#ifndef AXON_NEURALNETWORK_CUH
#define AXON_NEURALNETWORK_CUH

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

    class AXON_API NeuralNetwork
    {
    public:
        NeuralNetwork(const NeuralNetworkDescriptor& descriptor);
    private:

    };
}

#endif //AXON_NEURALNETWORK_CUH
