#ifndef AXON_BLOB_CUH
#define AXON_BLOB_CUH

#include <array>
#include <string>
#include <iostream>
#include <array>

#include <cuda_runtime.h>
#include <cudnn.h>

namespace axon
{
    typedef enum {
        host, cuda
    } DeviceType;

    template<typename ftype>
    class Blob
    {
    public:
        Blob(int n = 1, int c = 1, int h = 1, int w = 1)
            : n_(n), c_(c), h_(h), w_(w)
        {
            pHost_ = new float[n_ * c_ * h_ * w_];
        }

        Blob(const std::array<int, 4> shape)
            : n_(shape[0]), c_(shape[1]), h_(shape[2]), w_(shape[3])
        {
            pHost_ = new float[n_ * c_ * h_ * w_];
        }

        ~Blob() noexcept
        {
            if (pHost_ != nullptr)
                delete [] pHost_;
            if (pDevice_ != nullptr)
                cudaFree(pDevice_);
            if (is_tensor_)
                cudnnDestroyTensorDescriptor(tensor_desc_);
        }

        void reset(int n = 1, int c = 1, int h = 1, int w = 1)
        {
            n_ = n;
            c_ = c;
            h_ = h;
            w_ = w;

            if (pHost_ != nullptr)
            {
                delete [] pHost_;
                pHost_ = nullptr;
            }
            if (pDevice_ != nullptr)
            {
                cudaFree(pDevice_);
                pDevice_ = nullptr;
            }

            pHost_ = new float[n_ * c_ * h_ * w_];
            //  TODO: cuda()
            if (is_tensor_)
            {
                cudnnDestroyTensorDescriptor(tensor_desc_);
                is_tensor_ = false;
            }
        }

        void reset(std::array<int, 4> size)
        {
            reset(size[0], size[1], size[2], size[3]);
        }

        std::array<int, 4> shape() { return std::array<int, 4>({n_, c_, h_, w_}); }

        int size() { return c_ * h_ * w_; }
        int len() { return n_ * c_ * h_ * w_; }

        int byteWidth() { return sizeof(ftype) * len(); }

        [[nodiscard]] int n() const { return n_; }
        [[nodiscard]] int c() const { return c_; }
        [[nodiscard]] int h() const { return h_; }
        [[nodiscard]] int w() const { return w_; }

        cudnnTensorDescriptor_t tensor()
        {
            if (is_tensor_)
                return tensor_desc_;

            cudnnCreateTensorDescriptor(&tensor_desc_);
            cudnnSetTensor4dDescriptor(tensor_desc_,
                                       CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                       n_, c_, h_, w_);
            is_tensor_ = true;

            return tensor_desc_;
        }

        ftype *ptr() { return pHost_; }

        // get cuda memory
        ftype *cuda()
        {
            if (pDevice_ == nullptr)
                cudaMalloc((void**)&pDevice_, sizeof(ftype) * len());

            return pDevice_;
        }


        ftype *to(DeviceType target)
        {
            ftype *ptr = nullptr;
            if (target == host)
            {
                cudaMemcpy(pHost_, cuda(), sizeof(ftype) * len(), cudaMemcpyDeviceToHost);
                ptr = pHost_;
            }
            else // DeviceType::cuda
            {
                cudaMemcpy(cuda(), pHost_, sizeof(ftype) * len(), cudaMemcpyHostToDevice);
                ptr = pDevice_;
            }
            return ptr;
        }

        void print(std::string name, bool view_param = false, int num_batch = 1, int width = 16)
        {
            to(host);
            std::cout << "**" << name << "\t: (" << size() << ")\t";
            std::cout << ".n: " << n_ << ", .c: " << c_ << ", .h: " << h_ << ", .w: " << w_;
            std::cout << std::hex << "\t(h:" << pHost_ << ", d:" << pDevice_ << ")" << std::dec << std::endl;

            if (view_param)
            {
                std::cout << std::fixed;
                std::cout.precision(6);

                int max_print_line = 4;
                if (width == 28) {
                    std::cout.precision(3);
                    max_print_line = 28;
                }
                int offset = 0;

                for (int n = 0; n < num_batch; n++) {
                    if (num_batch > 1)
                        std::cout << "<--- batch[" << n << "] --->" << std::endl;
                    int count = 0;
                    int print_line_count = 0;
                    while (count < size() && print_line_count < max_print_line)
                    {
                        std::cout << "\t";
                        for (int s = 0; s < width && count < size(); s++)
                        {
                            std::cout << pHost_[size()*n + count + offset] << "\t";
                            count++;
                        }
                        std::cout << std::endl;
                        print_line_count++;
                    }
                }
                std::cout.unsetf(std::ios::fixed);
            }
        }

    private:
        bool is_tensor_ = false;
        cudnnTensorDescriptor_t tensor_desc_;

        ftype* pHost_ = nullptr;
        ftype* pDevice_ = nullptr;

        int n_ = 1;
        int c_ = 1;
        int h_ = 1;
        int w_ = 1;
    };

    using blob_f32 = Blob<float>;
}

#endif //AXON_BLOB_CUH
