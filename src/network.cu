#include "core/compute.cuh"
#include "core/network.cuh"

namespace axon
{
    neural_network::neural_network()
    {

    }

    neural_network::~neural_network()
    {
        for (auto& layer : _layers)
            delete layer;


        delete _cuda;
    }

    void neural_network::add_layer(layers::layer *layer)
    {
        _layers.push_back(layer);

        if (_layers.size() == 1)
            _layers.at(0)->set_gradient_stop();
    }

    blob_f32* neural_network::forward(blob_f32 *input)
    {
        _output = input;

        for (auto& layer : _layers)
        {
            _output = layer->forward(_output);
        }

        return _output;
    }

    void neural_network::backward(blob_f32 *input)
    {
        blob_f32* gradient = input;

        if (_phase == inference)
            return;

        for (auto layer = _layers.rbegin(); layer != _layers.rend(); layer++)
        {
            gradient = (*layer)->backward(gradient);
        }
    }

    void neural_network::update(float learningRate)
    {
        if (_phase == inference) return;

        for (auto &layer: _layers)
        {
            if (!layer->has_parameters()) continue;
            layer->update_weights_biases(learningRate);
        }
    }

    void neural_network::cuda()
    {
        _cuda = new cuda_context();
        for (auto& layer : _layers)
            layer->set_cuda_context(_cuda);
    }

    void neural_network::train()
    {
        _phase = training;

        for (auto& layer : _layers)
            layer->unfreeze();
    }

    void neural_network::test()
    {
        _phase = inference;

        for (auto& layer : _layers)
            layer->freeze();
    }

    float neural_network::loss(blob_f32 *target)
    {
        auto layer = _layers.back();
        return layer->get_loss(target);
    }

    int neural_network::get_accuracy(blob_f32 *target)
    {
        auto layer = _layers.back();
        return layer->get_accuracy(target);
    }
}

