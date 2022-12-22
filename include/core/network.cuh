#ifndef AXON_NETWORK_CUH
#define AXON_NETWORK_CUH

#include "core/blob.cuh"
#include "core/compute.cuh"
#include "core/layer.cuh"

#include <vector>

namespace axon
{
    typedef enum {
        training, inference
    } workload_type_t;

    class neural_network
    {
    public:
        neural_network();
        ~neural_network();

        void add_layer(layers::layer* layer);

        blob_f32* forward(blob_f32* input);
        void backward(blob_f32* input = nullptr);
        void update(float learningRate = 0.02f);

        float loss(blob_f32* target);
        int get_accuracy(blob_f32* target);

        void cuda();
        void train();
        void test();

        const std::vector<layers::layer*>& layers() const { return _layers; }
    private:
        cuda_context *_cuda = nullptr;
        std::vector<layers::layer*> _layers;

        blob_f32* _output;
        workload_type_t _phase = inference;
    };
}

#endif //AXON_NETWORK_CUH
