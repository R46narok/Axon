#include "core/blob.cuh"
#include "core/mnist.cuh"
#include "core/network.cuh"
#include <iomanip>
using namespace axon;
using namespace axon::layers;

int main()
{
    int batch_size_train = 256;
    int num_steps_train = 2400;
    int monitoring_step = 200;

    double learning_rate = 0.02f;
    double lr_decay = 0.00005f;

    bool load_pretrain = false;
    bool file_save = false;

    int batch_size_test = 10;
    int num_steps_test = 1000;


    /* Welcome Message */
    std::cout << "== MNIST training with CUDNN ==" << std::endl;

    // phase 1. training
    std::cout << "[TRAIN]" << std::endl;

    // step 1. loading dataset
    MNIST train_data_loader = MNIST("./dataset");
    train_data_loader.train(batch_size_train, true);

    // step 2. model initialization
    neural_network model;
    model.add_layer(new conv2d("conv1", 20, 5));
    model.add_layer(new pooling("pool", 2, 0, 2, CUDNN_POOLING_MAX));
    model.add_layer(new conv2d("conv2", 50, 5));
    model.add_layer(new pooling("pool", 2, 0, 2, CUDNN_POOLING_MAX));
    model.add_layer(new dense("dense1", 700));
    model.add_layer(new activation("relu", CUDNN_ACTIVATION_RELU));
    model.add_layer(new dense("dense2", 700));
    model.add_layer(new activation("relu1", CUDNN_ACTIVATION_RELU));
    model.add_layer(new dense("dense3", 10));
    model.add_layer(new softmax("softmax"));
    model.cuda();

    model.train();

    // step 3. train
    int step = 0;
    Blob<float> *train_data = train_data_loader.get_data();
    Blob<float> *train_target = train_data_loader.get_target();
    train_data_loader.get_batch();
    int tp_count = 0;
    while (step < num_steps_train)
    {
        // nvtx profiling start
        std::string nvtx_message = std::string("step" + std::to_string(step));

        // update shared buffer contents
        train_data->to(cuda);
        train_target->to(cuda);

        // forward
        model.forward(train_data);
        tp_count += model.get_accuracy(train_target);

        // back-propagation
        model.backward(train_target);

        // update parameter
        // we will use learning rate decay to the learning rate
        learning_rate *= 1.f / (1.f + lr_decay * step);
        model.update(learning_rate);

        // fetch next data
        step = train_data_loader.next();

        // nvtx profiling end

        // calculation softmax loss
        if (step % monitoring_step == 0)
        {
            float loss = model.loss(train_target);
            float accuracy =  100.f * tp_count / monitoring_step / batch_size_train;

            std::cout << "step: " << std::right << std::setw(4) << step << \
                         ", loss: " << std::left << std::setw(5) << std::fixed << std::setprecision(3) << loss << \
                         ", accuracy: " << accuracy << "%" << std::endl;

            tp_count = 0;
        }
    }

    // trained parameter save

    // phase 2. inferencing
    // step 1. load test set
    std::cout << "[INFERENCE]" << std::endl;
    MNIST test_data_loader = MNIST("./dataset");
    test_data_loader.test(batch_size_test);

    // step 2. model initialization
    model.test();

    // step 3. iterates the testing loop
    Blob<float> *test_data = test_data_loader.get_data();
    Blob<float> *test_target = test_data_loader.get_target();
    test_data_loader.get_batch();
    tp_count = 0;
    step = 0;
    while (step < num_steps_test)
    {
        // nvtx profiling start
        std::string nvtx_message = std::string("step" + std::to_string(step));

        // update shared buffer contents
        test_data->to(cuda);
        test_target->to(cuda);

        // forward
        model.forward(test_data);
        tp_count += model.get_accuracy(test_target);

        // fetch next data
        step = test_data_loader.next();

        // nvtx profiling stop
    }

    // step 4. calculate loss and accuracy
    float loss = model.loss(test_target);
    float accuracy = 100.f * tp_count / num_steps_test / batch_size_test;

    std::cout << "loss: " << std::setw(4) << loss << ", accuracy: " << accuracy << "%" << std::endl;

    // Good bye
    std::cout << "Done." << std::endl;

    return 0;
}