#ifndef AXON_LOSS_CUH
#define AXON_LOSS_CUH

#include "core/blob.cuh"

namespace axon
{
    class cross_entropy_loss
    {
    public:
        cross_entropy_loss();
        ~cross_entropy_loss();

        float loss(blob_f32 * predict, blob_f32 * target);
        float accuracy(blob_f32 * predict, blob_f32 * target);

    private:
        float hostLoss_ = 0.0f;
        float* pDeviceLoss_ = nullptr;

        float* pDeviceWorkspace_ = nullptr;
        void initWorkspace(int batchSize);
    };
}

#endif //AXON_LOSS_CUH
