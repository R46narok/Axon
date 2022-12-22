#ifndef AXON_LAYER_CUH
#define AXON_LAYER_CUH

#include "core/compute.cuh"
#include "core/blob.cuh"
#include "core/loss.cuh"

namespace axon::layers
{
    class layer
    {
    public:
        layer() = default;
         ~layer();

        virtual blob_f32* forward(blob_f32* input) = 0;
        virtual blob_f32* backward(blob_f32* gradInput) = 0;

        [[nodiscard]] const std::string& get_name() const { return _name; }

        virtual float get_loss(blob_f32* target);
        virtual int get_accuracy(blob_f32* target);

        void set_cuda_context(cuda_context* context) { _pCuda = context; }
        void set_gradient_stop() { _gradientStop = true; }

        void freeze() { _freeze = true; }
        void unfreeze() { _freeze = false; }

        bool has_parameters()
        {
            return  _pWeights != nullptr && _pGradWeights != nullptr &&
                    _pBiases != nullptr && _pGradBiases != nullptr;
        }
        void update_weights_biases(float learningRate);
    protected:
        std::string _name;

        cudnnTensorDescriptor_t _inputDesc;
        cudnnTensorDescriptor_t _outputDesc;

        cudnnFilterDescriptor_t _filterDesc;
        cudnnTensorDescriptor_t _biasDesc;

        blob_f32* _pInput = nullptr;
        blob_f32* _pOutput = nullptr;
        blob_f32* _pGradInput = nullptr;
        blob_f32* _pGradOutput = nullptr;

        bool _freeze = false;
        blob_f32* _pWeights = nullptr;
        blob_f32* _pBiases = nullptr;
        blob_f32* _pGradWeights = nullptr;
        blob_f32* _pGradBiases = nullptr;

        int _batchSize = 0;

        void init_weights_bias(uint32_t seed = 0);

        cuda_context* _pCuda = nullptr;

        bool _gradientStop = false;
    };

    class dense : public layer
    {
    public:
        dense(const std::string& name, int outSize);
        ~dense();

        blob_f32* forward(blob_f32* input) override;
        blob_f32* backward(blob_f32* gradInput) override;

    private:
        int _inputSize = 0;
        int _outputSize = 0;

        float* _pDeviceOneVec = nullptr;
    };

    class activation : public layer
    {
    public:
        activation(const std::string& name, cudnnActivationMode_t mode, float coef = 0.0f);
        ~activation();

        blob_f32* forward(blob_f32* input) override;
        blob_f32* backward(blob_f32* gradOutput) override;
    private:
        cudnnActivationDescriptor_t _desc;
        cudnnActivationMode_t _mode;
        float _coef;
    };

    class softmax : public layer
    {
    public:
        softmax(const std::string& name);
        ~softmax();

        blob_f32* forward(blob_f32 *input) override;
        blob_f32* backward(blob_f32 *gradInput) override;

        float get_loss(blob_f32 *target) override;
        int get_accuracy(blob_f32 *target) override;
    private:
        cross_entropy_loss _loss;
    };

    class conv2d: public layer
    {
    public:
        conv2d(const std::string& name,
               int out_channels,
               int kernel_size,
               int stride=1,
               int padding=0,
               int dilation=1);
        ~conv2d();

        blob_f32 *forward(blob_f32 *input) override;
        blob_f32 *backward(blob_f32 *grad_output) override;

    private:
        int _outChannels;
        int _kernelSize;
        int _stride;
        int _padding;
        int _dilation;

        std::array<int, 4> _outputSize;

        // convolution
        cudnnConvolutionDescriptor_t    _convDesc;

        cudnnConvolutionFwdAlgo_t       _convFwdAlgo;
        cudnnConvolutionBwdDataAlgo_t   _convBwdDataAlgo;
        cudnnConvolutionBwdFilterAlgo_t _convBwdFilterAlgo;

        size_t workspace_size = 0;
        void** d_workspace = nullptr;
        void set_workspace();
    };

    class pooling: public layer
    {
    public:
        pooling(const std::string& name,
                int kernel_size,
                int padding,
                int stride,
                cudnnPoolingMode_t mode);
        ~pooling();

        blob_f32 *forward(blob_f32 *input) override;
        blob_f32 *backward(blob_f32 *grad_output) override;

    private:
        int _kernelSize;
        int _padding;
        int _stride;
        cudnnPoolingMode_t       _mode;

        std::array<int, 4> _outputSize;
        cudnnPoolingDescriptor_t _poolDesc;
    };
}

#endif //AXON_LAYER_CUH
