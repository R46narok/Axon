#include "core/layer.cuh"
#include "core/compute.cuh"

#include <random>

#include <cuda_runtime.h>
#include <curand.h>
#include <cassert>
#include <math.h>
#include <algorithm>

#include <iostream>

namespace axon::layers
{

    layer::~layer()
    {
#if (DEBUG_FORWARD > 0 || DEBUG_BACKWARD > 0)
        std::cout << "Destroy layer: " << _name << std::endl;
#endif

        if (_pOutput       != nullptr)  delete _pOutput;
        if (_pGradInput   != nullptr)  delete _pGradInput;

        if (_pWeights      != nullptr)  delete _pWeights;
        if (_pBiases       != nullptr)  delete _pBiases;
        if (_pGradWeights != nullptr)  delete _pGradWeights;
        if (_pGradBiases  != nullptr)  delete _pGradBiases;
    }

    void layer::init_weights_bias(unsigned int seed)
    {
        checkCudaErrors(cudaDeviceSynchronize());

        if (_pWeights == nullptr || _pBiases == nullptr)
            return;

        // Create random network
        std::random_device rd;
        std::mt19937 gen(seed == 0 ? rd() : static_cast<unsigned int>(seed));

        // He uniform distribution
        float range = sqrt(6.f / _pInput->size());	// He's initialization
        std::uniform_real_distribution<> dis(-range, range);

        for (int i = 0; i < _pWeights->len(); i++)
            _pWeights->ptr()[i] = static_cast<float>(dis(gen));
        for (int i = 0; i < _pBiases->len(); i++)
            _pBiases->ptr()[i] = 0.f;

        // copy initialized value to the device
        _pWeights->to(DeviceType::cuda);
        _pBiases->to(DeviceType::cuda);

        std::cout << ".. initialized " << _name << " layer .." << std::endl;
    }

    void layer::update_weights_biases(float learning_rate)
    {
        float eps = -1.f * learning_rate;
        if (_pWeights != nullptr && _pGradWeights != nullptr)
        {
#if (DEBUG_UPDATE)
            _pWeights->print(_name + "::weights (before update)", true);
		_pGradWeights->print(_name + "::gweights", true);
#endif // DEBUG_UPDATE

            // w = w + eps * dw
            checkCublasErrors(
                    cublasSaxpy(_pCuda->cublas(),
                                _pWeights->len(),
                                &eps,
                                _pGradWeights->cuda(), 1,
                                _pWeights->cuda(), 1));

#if (DEBUG_UPDATE)
            _pWeights->print(_name + "weights (after update)", true);
		// getchar();
#endif // DEBUG_UPDATE
        }

        if (_pBiases != nullptr && _pGradBiases != nullptr)
        {
#if (DEBUG_UPDATE)
            _pBiases->print(_name + "biases (before update)", true);
		_pGradBiases->print(_name + "gbiases", true);
#endif // DEBUG_UPDATE

            // b = b + eps * db
            checkCublasErrors(
                    cublasSaxpy(_pCuda->cublas(),
                                _pBiases->len(),
                                &eps,
                                _pGradBiases->cuda(), 1,
                                _pBiases->cuda(), 1));

#if (DEBUG_UPDATE)
            _pBiases->print(_name + "biases (after update)", true);
		// getchar();
#endif // DEBUG_UPDATE
        }
    }

    float layer::get_loss(Blob<float> *target)
    {
        assert("No Loss layer has no loss." && false);
        return EXIT_FAILURE;
    }

    int layer::get_accuracy(Blob<float> *target)
    {
        assert("No Loss layer cannot estimate accuracy." && false);
        return EXIT_FAILURE;
    }



/****************************************************************
 * dense layer                                                  *
 ****************************************************************/

    dense::dense(const std::string& name, int output_size)
    {
        _name = name;
        _outputSize = output_size;
    }

    dense::~dense()
    {
        if (_pDeviceOneVec != nullptr)
            cudaFree(_pDeviceOneVec);
    }

    __global__ void init_one_vec(float* _pDeviceOneVec, size_t length)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        if (i >= length) return;

        _pDeviceOneVec[i] = 1.f;
    }

    Blob<float> *dense::forward(Blob<float> *input)
    {
        // initialize weights and biases
        if (_pWeights == nullptr)
        {
            // setup parameter size information
            _inputSize  = input->c() * input->h() * input->w();

            // initialize weight, bias, and output
            _pWeights = new Blob<float>(1, 1, _inputSize, _outputSize);
            _pBiases  = new Blob<float>(1, 1, _outputSize);

        }

        // initilaize input and output
        if (_pInput == nullptr || _batchSize != input->n())
        {
            _pInput = input;
            _batchSize  = input->n();

            if (_pOutput == nullptr)
                _pOutput  = new Blob<float>(_batchSize, _outputSize);
            else
                _pOutput->reset(_batchSize, _outputSize);

            _pOutput->tensor();

            if (_pDeviceOneVec != nullptr)
                cudaFree(_pDeviceOneVec);
            checkCudaErrors(cudaMalloc((void**)&_pDeviceOneVec, sizeof(float) * _batchSize));
            init_one_vec<<< (_batchSize+BLOCK_DIM_1D-1)/BLOCK_DIM_1D, BLOCK_DIM_1D >>>(_pDeviceOneVec, _batchSize);

            // initialize weights and biases

            if (!_freeze)
            {
                init_weights_bias();
            }
            else
            {
                /* do nothing */
            }
        }


        // output = weights^T * input (without biases)
        checkCublasErrors(
                cublasSgemm(_pCuda->cublas(),
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            _outputSize, _batchSize, _inputSize,
                            &_pCuda->one,
                            _pWeights->cuda(), _inputSize,
                            _pInput->cuda(), _inputSize,
                            &_pCuda->zero,
                            _pOutput->cuda(),  _outputSize));

        // output += biases * _pDeviceOneVec^T
        checkCublasErrors(cublasSgemm(_pCuda->cublas(),
                                      CUBLAS_OP_N, CUBLAS_OP_N,
                                      _outputSize, _batchSize, 1,
                                      &_pCuda->one,
                                      _pBiases->cuda(), _outputSize,
                                      _pDeviceOneVec, 1,
                                      &_pCuda->one,
                                      _pOutput->cuda(), _outputSize));

#if (DEBUG_DENSE & 0x01)
        _pInput->print(  _name + "::input",  true);
	_pWeights->print(_name + "::weight", true);
	_pBiases->print( _name + "::bias",   true);
	_pOutput->print( _name + "::output", true);
#endif // DEBUG_DENSE

        return _pOutput;
    }

    Blob<float> *dense::backward(Blob<float> *grad_output)
    {
        if (_pGradWeights == nullptr)
        {
            _pGradWeights = new Blob<float>(_pWeights->shape());
            _pGradBiases  = new Blob<float>(_pBiases->shape());
        }

        if (_pGradInput == nullptr || _batchSize != grad_output->n())
        {
            _pGradOutput  = grad_output;

            if (_pGradInput == nullptr)
                _pGradInput   = new Blob<float>(_pInput->shape());
            else
                _pGradInput->reset(_pInput->shape());
        }

        // db = (dy) * _pDeviceOneVec
        cublasSgemv(_pCuda->cublas(),
                    CUBLAS_OP_N,
                    _outputSize, _batchSize,
                    &_pCuda->one,
                    _pGradOutput->cuda(), _outputSize,
                    _pDeviceOneVec, 1,
                    &_pCuda->zero,
                    _pGradBiases->cuda(), 1);

        // dw = x * (dy)^T
        cublasSgemm(_pCuda->cublas(),
                    CUBLAS_OP_N, CUBLAS_OP_T,
                    _inputSize, _outputSize, _batchSize,
                    &_pCuda->one,
                    _pInput->cuda(),        _inputSize,
                    _pGradOutput->cuda(),  _outputSize,
                    &_pCuda->zero,
                    _pGradWeights->cuda(), _inputSize);

        // dx = W * dy
        if (!_gradientStop)
            cublasSgemm(_pCuda->cublas(),
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        _inputSize, _batchSize, _outputSize,
                        &_pCuda->one,
                        _pWeights->cuda(),     _inputSize,
                        _pGradOutput->cuda(), _outputSize,
                        &_pCuda->zero,
                        _pGradInput->cuda(),  _inputSize);

#if (DEBUG_DENSE & 0x02)
        std::cout << _name << "[BACKWARD]" << std::endl;
	grad_output->print(  _name + "::gradients", true, grad_output->n());
	_pGradWeights->print(_name + "::gfilter", true);
	_pGradBiases->print( _name + "::gbias", true);
	if (!_gradientStop)
		_pGradInput->print(  _name + "::gdata", true);
#endif // DEBUG_DENSE

        return _pGradInput;
    }

/****************************************************************
 * activation layer                                             *
 ****************************************************************/

    activation::activation(const std::string& name, cudnnActivationMode_t mode, float coef)
    {
        _name = name;
        _mode = mode;
        _coef = coef;

        cudnnCreateActivationDescriptor(&_desc);
        cudnnSetActivationDescriptor(_desc, mode, CUDNN_PROPAGATE_NAN, coef);
    }

    activation::~activation()
    {
        cudnnDestroyActivationDescriptor(_desc);
    }

    Blob<float> *activation::forward(Blob<float> *input)
    {
        if (_pInput == nullptr || _batchSize != input->n())
        {
            _pInput = input;
            _inputDesc = input->tensor();
            _batchSize  = input->n();

            if (_pOutput == nullptr)
                _pOutput = new Blob<float>(input->shape());
            else
                _pOutput->reset(input->shape());

            _outputDesc = _pOutput->tensor();
        }

        cudnnActivationForward(_pCuda->cudnn(),
                               _desc,
                               &_pCuda->one,
                               _inputDesc,
                               input->cuda(),
                               &_pCuda->zero,
                               _outputDesc,
                               _pOutput->cuda());

        return _pOutput;
    }

    Blob<float> *activation::backward(Blob<float> *grad_output)
    {
        if (_pGradInput == nullptr || _batchSize != grad_output->n())
        {
            _pGradOutput = grad_output;

            if (_pGradInput == nullptr)
                _pGradInput = new Blob<float>(_pInput->shape());
            else
                _pGradInput->reset(_pInput->shape());
        }

        cudnnActivationBackward(_pCuda->cudnn(),
                                _desc,
                                &_pCuda->one,
                                _outputDesc, _pOutput->cuda(),
                                _outputDesc, grad_output->cuda(),
                                _inputDesc, _pInput->cuda(),
                                &_pCuda->zero,
                                _inputDesc, _pGradInput->cuda());

        return _pGradInput;
    }

/****************************************************************
 * softmax definition                                           *
 ****************************************************************/

    softmax::softmax(const std::string& name)
    {
        _name = name;
    }

    softmax::~softmax()
    {

    }

    Blob<float> *softmax::forward(Blob<float> *input)
    {
        if (_pInput == nullptr || _batchSize != input->n())
        {
            _pInput = input;
            _inputDesc = input->tensor();
            _batchSize  = input->n();

            if (_pOutput == nullptr)
                _pOutput = new Blob<float>(input->shape());
            else
                _pOutput->reset(input->shape());

            _outputDesc = _pOutput->tensor();
        }

#if (DEBUG_SOFTMAX & 0x01)
            std::cout << _name << "[FORWARD]" << std::endl;
	_pInput->print(_name + "::input", true, input->n());
#endif

        checkCudnnErrors(
                cudnnSoftmaxForward(_pCuda->cudnn(), CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
                                    &_pCuda->one,  _inputDesc,  input->cuda(),
                                    &_pCuda->zero, _outputDesc, _pOutput->cuda()));

#if (DEBUG_SOFTMAX & 0x01)
        _pOutput->print(_name + "::output", true, input->n());
#endif

        return _pOutput;
    }

    Blob<float> *softmax::backward(Blob<float> *target)
    {
        checkCudaErrors(cudaDeviceSynchronize());

        if (_pGradInput == nullptr || _batchSize != target->n())
        {
            if (_pGradInput == nullptr)
                _pGradInput = new Blob<float>(_pInput->shape());
            else
                _pGradInput->reset(_pInput->shape());
        }

        // set _pGradInput as predict
        checkCudaErrors(cudaMemcpyAsync(_pGradInput->cuda(),
                                        _pOutput->cuda(), _pOutput->byteWidth(),
                                        cudaMemcpyDeviceToDevice));
        // set _pGradInput = predict - target
        checkCublasErrors(
                cublasSaxpy(_pCuda->cublas(), target->len(),
                            &_pCuda->minus_one, target->cuda(), 1,
                            _pGradInput->cuda(), 1));

        // normalize the grad_output by the batch size
        int grad_output_size = target->n() * target->c() * target->h() * target->w();
        float scale = 1.f / static_cast<float>(target->n());
        checkCublasErrors(cublasSscal(_pCuda->cublas(), grad_output_size, &scale, _pGradInput->cuda(), 1));

#if (DEBUG_SOFTMAX & 0x02)
        std::cout << _name << "[BACKWARD]" << std::endl;
	_pInput->print( _name + "::input", true);
	_pOutput->print(_name + "::predict", true);
	target->print( _name + "::y", true, target->n());
	_pGradInput->print(_name + "::dx", true, target->n());
#endif

        return _pGradInput;
    }

    float softmax::get_loss(Blob<float> *target)
    {
        return _loss.loss(_pOutput, target);
    }

    int softmax::get_accuracy(Blob<float> *target)
    {
        int batch_size = _pOutput->n();
        int output_size = _pOutput->size();

        assert(batch_size == target->n());
        assert(output_size == target->size());

        float *h_output, *h_target;
        int idx_output, idx_target;
        int hit_count = 0;

        // get predicts and targets
        h_output = _pOutput->to(host);
        h_target = target->to(host);

        // idx_output = idx_target = 0;
        for (int b = 0; b < batch_size; b++)
        {
            idx_output = 0;
            idx_target = 0;

            for (int i = 1; i < 10; i++)
            {
                if (h_output[b * output_size + i] > h_output[b * output_size + idx_output])
                    idx_output = i;
                if (h_target[b * output_size + i] > h_target[b * output_size + idx_target])
                    idx_target = i;
            }

            if (idx_output == idx_target)
                hit_count++;
        }

        return hit_count;
    }

/****************************************************************
 * layer definition                                             *
 ****************************************************************/

/**
 * Convolutional layer with bias
 */
    conv2d::conv2d(const std::string& name,
                   int out_channels,
                   int kernel_size,
                   int stride,
                   int padding,
                   int dilation):
            _outChannels(out_channels),
            _kernelSize(kernel_size),
            _stride(stride),
            _padding(padding),
            _dilation(dilation)
    {
        _name = name;

        // create cudnn container handles
        cudnnCreateFilterDescriptor(&_filterDesc);

        cudnnCreateConvolutionDescriptor(&_convDesc);
        checkCudnnErrors(cudnnSetConvolution2dDescriptor(_convDesc,
                                                         _padding, _padding, _stride,  _stride, _dilation,_dilation,
                                                         CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

        d_workspace = nullptr;
    }

    conv2d::~conv2d()
    {
        // distroy cudnn container resources
        cudnnDestroyFilterDescriptor(_filterDesc);
        cudnnDestroyConvolutionDescriptor(_convDesc);

        // terminate internal created blobs
        if (d_workspace != nullptr)	cudaFree(d_workspace);
    }

    void conv2d::set_workspace()
    {
        size_t temp_size = 0;

        cudnnConvolutionFwdAlgoPerf_t 		fwd_algo_perf_results[CUDNN_CONVOLUTION_FWD_ALGO_COUNT];
        cudnnConvolutionBwdFilterAlgoPerf_t 	bwd_filter_algo_perf_results[CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT];
        cudnnConvolutionBwdDataAlgoPerf_t	bwd_data_algo_perf_results[CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT];

        // forward
#if CUDNN_MAJOR == 8
        int algo_max_count;
        checkCudnnErrors(cudnnGetConvolutionForwardAlgorithmMaxCount(_pCuda->cudnn(), &algo_max_count));
        std::cout << this->_name << ": Available Algorithm Count [FWD]: " << algo_max_count << std::endl;
        checkCudnnErrors(cudnnGetConvolutionForwardAlgorithm_v7(_pCuda->cudnn(),
                                                                _inputDesc, _filterDesc, _convDesc, _outputDesc,
                                                                algo_max_count, 0, fwd_algo_perf_results));
        _convFwdAlgo = fwd_algo_perf_results[0].algo;
#else
        checkCudnnErrors(cudnnGetConvolutionForwardAlgorithm(_pCuda->cudnn(),
		_inputDesc, _filterDesc, _convDesc, _outputDesc,
		CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &_convFwdAlgo));
#endif
        checkCudnnErrors(cudnnGetConvolutionForwardWorkspaceSize(_pCuda->cudnn(),
                                                                 _inputDesc, _filterDesc, _convDesc, _outputDesc,
                                                                 _convFwdAlgo, &temp_size));
        workspace_size = std::max(workspace_size, temp_size);

        // bwd - filter
#if CUDNN_MAJOR == 8
        checkCudnnErrors(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(_pCuda->cudnn(), &algo_max_count));
        std::cout << this->_name << ": Available Algorithm Count [BWD-filter]: " << algo_max_count << std::endl;
        checkCudnnErrors(cudnnGetConvolutionBackwardFilterAlgorithm_v7(_pCuda->cudnn(),
                                                                       _inputDesc, _outputDesc, _convDesc, _filterDesc,
                                                                       algo_max_count, 0, bwd_filter_algo_perf_results));
        _convBwdFilterAlgo = bwd_filter_algo_perf_results[0].algo;
#else
        checkCudnnErrors(cudnnGetConvolutionBackwardFilterAlgorithm(_pCuda->cudnn(),
		_inputDesc, _outputDesc, _convDesc, _filterDesc,
		CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &_convBwdFilterAlgo));
#endif
        checkCudnnErrors(cudnnGetConvolutionBackwardFilterWorkspaceSize(_pCuda->cudnn(),
                                                                        _inputDesc, _outputDesc, _convDesc, _filterDesc,
                                                                        _convBwdFilterAlgo, &temp_size));
        workspace_size = std::max(workspace_size, temp_size);

        // bwd - data
#if CUDNN_MAJOR == 8
        checkCudnnErrors(cudnnGetConvolutionBackwardDataAlgorithmMaxCount(_pCuda->cudnn(), &algo_max_count));
        std::cout << this->_name << ": Available Algorithm Count [BWD-data]: " << algo_max_count << std::endl;
        checkCudnnErrors(cudnnGetConvolutionBackwardDataAlgorithm_v7(_pCuda->cudnn(),
                                                                     _filterDesc, _outputDesc, _convDesc, _inputDesc,
                                                                     algo_max_count, 0, bwd_data_algo_perf_results));
        _convBwdDataAlgo = bwd_data_algo_perf_results[0].algo;
#else
        checkCudnnErrors(cudnnGetConvolutionBackwardDataAlgorithm(_pCuda->cudnn(),
		_filterDesc, _outputDesc, _convDesc, _inputDesc,
		CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &_convBwdDataAlgo));
#endif
        checkCudnnErrors(cudnnGetConvolutionBackwardDataWorkspaceSize(_pCuda->cudnn(),
                                                                      _filterDesc, _outputDesc, _convDesc, _inputDesc,
                                                                      _convBwdDataAlgo, &temp_size));
        workspace_size = std::max(workspace_size, temp_size);

        if (workspace_size > 0)
        {
            if (d_workspace != nullptr)
            checkCudaErrors(cudaFree(d_workspace));
            checkCudaErrors(cudaMalloc((void**)&d_workspace, workspace_size));
        }
    }

    Blob<float> *conv2d::forward(Blob<float> *input)
    {
        // initialize weights and bias
        if (_pWeights == nullptr)
        {
            // initialize containers handles
            checkCudnnErrors(cudnnSetFilter4dDescriptor(_filterDesc,
                                                        CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                                        _outChannels, input->c(), _kernelSize, _kernelSize));

            _pWeights = new Blob<float>(_outChannels, input->c(), _kernelSize, _kernelSize);
            _pBiases  = new Blob<float>(1, _outChannels);	// bias size
            _biasDesc = _pBiases->tensor();
        }

        // initilaize input and output
        if (_pInput == nullptr || _batchSize != input->n())
        {
            // initialize input
            _pInput = input;
            _inputDesc = input->tensor();
            _batchSize  = input->n();

            // initilaize output
            checkCudnnErrors(cudnnGetConvolution2dForwardOutputDim(
                    _convDesc, _inputDesc, _filterDesc,
                    &_outputSize[0], &_outputSize[1], &_outputSize[2], &_outputSize[3]));

            if (_pOutput == nullptr)
                _pOutput  = new Blob<float>(_outputSize);
            else
                _pOutput->reset(_outputSize);

            _outputDesc = _pOutput->tensor();

            // initialize workspace for cudnn
            set_workspace();

            // initialize weights

            if (!_freeze)
            {
                init_weights_bias();
            }
            else
            {
                /* do nothing */
            }
        }

        checkCudnnErrors(cudnnConvolutionForward(_pCuda->cudnn(),
                                                 &_pCuda->one,  _inputDesc,  _pInput->cuda(),
                                                 _filterDesc, _pWeights->cuda(), _convDesc, _convFwdAlgo, d_workspace,  workspace_size,
                                                 &_pCuda->zero, _outputDesc, _pOutput->cuda()));

        checkCudnnErrors(cudnnAddTensor(_pCuda->cudnn(),
                                        &_pCuda->one, _biasDesc, _pBiases->cuda(),
                                        &_pCuda->one, _outputDesc, _pOutput->cuda()));

#if (DEBUG_CONV & 0x01)
        _pInput->print(  _name + "::input", true, _pInput->n(), 28);
	_pWeights->print(_name + "::weight", true);
	_pBiases->print( _name + "::bias", true);
	_pOutput->print( _name + "::output", true);
#endif

        return _pOutput;
    }

    Blob<float> *conv2d::backward(Blob<float> *grad_output)
    {
        // initialize grad_output back-propagation space
        if (_pGradInput == nullptr || _batchSize != grad_output->n()) {
            _pGradOutput  = grad_output;
            _pGradWeights = new Blob<float>(_pWeights->shape());
            _pGradBiases  = new Blob<float>(1, _pBiases->c());

            if (_pGradInput == nullptr)
                _pGradInput = new Blob<float>(_pInput->shape());
            else
                _pGradInput->reset(_pInput->shape());
        }

        // gradients of biases
        checkCudnnErrors(
                cudnnConvolutionBackwardBias(_pCuda->cudnn(),
                                             &_pCuda->one,
                                             _outputDesc, grad_output->cuda(),
                                             &_pCuda->zero,
                                             _biasDesc,   _pGradBiases->cuda()));

        // gradients of weights
        checkCudnnErrors(
                cudnnConvolutionBackwardFilter(_pCuda->cudnn(),
                                               &_pCuda->one,
                                               _inputDesc, _pInput->cuda(),
                                               _outputDesc, _pGradOutput->cuda(),
                                               _convDesc, _convBwdFilterAlgo, d_workspace, workspace_size,
                                               &_pCuda->zero,
                                               _filterDesc, _pGradWeights->cuda()));

        // gradients of input data
        if (!_gradientStop)
        checkCudnnErrors(
                cudnnConvolutionBackwardData(_pCuda->cudnn(),
                                             &_pCuda->one,
                                             _filterDesc, _pWeights->cuda(),
                                             _outputDesc, grad_output->cuda(),
                                             _convDesc, _convBwdDataAlgo, d_workspace, workspace_size,
                                             &_pCuda->zero,
                                             _inputDesc, _pGradInput->cuda()));

#if (DEBUG_CONV & 0x02)
        std::cout << _name << "[BACKWARD]" << std::endl;
	grad_output->print( _name + "::gradients", true);
	_pGradBiases->print(_name + "gbias", true);
	_pGradWeights->print(_name+ "gfilter", true);
	if (!_gradientStop)
		_pGradInput->print(_name+"gdata", true);
#endif

#if (DEBUG_CONV & 0x04)
        grad_output->print( _name + "::gradients", true);
	_pGradBiases->print( _name + "::gbias", true);
#endif

        return _pGradInput;
    }

/****************************************************************
 * layer definition                                             *
 ****************************************************************/

    pooling::pooling(const std::string& name,
                     int kernel_size,
                     int padding,
                     int stride,
                     cudnnPoolingMode_t mode):
            _kernelSize(kernel_size),
            _padding(padding),
            _stride(stride),
            _mode(mode)
    {
        _name = name;

        cudnnCreatePoolingDescriptor(&_poolDesc);
        cudnnSetPooling2dDescriptor(_poolDesc, _mode, CUDNN_PROPAGATE_NAN,
                                    _kernelSize, _kernelSize, _padding, _padding, _stride, _stride);
    }

    pooling::~pooling()
    {
        cudnnDestroyPoolingDescriptor(_poolDesc);
    }

    Blob<float> *pooling::forward(Blob<float> *input)
    {
        if (_pInput == nullptr || _batchSize != input->n())
        {
            _pInput = input;

            // resource initialize
            _inputDesc = _pInput->tensor();
            _batchSize  = input->n();

            // setting output
            cudnnGetPooling2dForwardOutputDim(_poolDesc, _inputDesc,
                                              &_outputSize[0], &_outputSize[1], &_outputSize[2], &_outputSize[3]);
            if (_pOutput == nullptr)
                _pOutput = new Blob<float>(_outputSize);
            else
                _pOutput->reset(_outputSize);

            _outputDesc = _pOutput->tensor();
        }

        cudnnPoolingForward(_pCuda->cudnn(), _poolDesc,
                            &_pCuda->one,   _inputDesc,  _pInput->cuda(),
                            &_pCuda->zero,  _outputDesc, _pOutput->cuda());

        return _pOutput;
    }

    Blob<float> *pooling::backward(Blob<float> *grad_output)
    {
        if (_pGradInput == nullptr || _batchSize != grad_output->n())
        {
            _pGradOutput = grad_output;

            if (_pGradInput == nullptr)
                _pGradInput = new Blob<float>(_pInput->shape());
            else
                _pGradInput->reset(_pInput->shape());
        }

        checkCudnnErrors(
                cudnnPoolingBackward(_pCuda->cudnn(), _poolDesc,
                                     &_pCuda->one,
                                     _outputDesc, _pOutput->cuda(),
                                     _outputDesc, grad_output->cuda(),
                                     _inputDesc,  _pInput->cuda(),
                                     &_pCuda->zero,
                                     _inputDesc,  _pGradInput->cuda()));

        return _pGradInput;
    }
}