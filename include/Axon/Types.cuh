//
// Created by r46narok on 25.03.22 г..
//

#ifndef AXON_TYPES_CUH
#define AXON_TYPES_CUH

#include <memory>
#include <cstdint>

namespace Axon
{
    class Buffer
    {
    public:
        Buffer() = default;

        template<class T>
        [[nodiscard]] static std::shared_ptr<Buffer> AllocateDeviceMemory(size_t size = sizeof(T))
        {
            auto ptr = CreateBuffer();
            cudaMalloc(&ptr->m_pDeviceBuffer, size);
            return ptr;
        }

        [[nodiscard]] static std::shared_ptr<Buffer> AllocateDeviceMemory(size_t size)
        {
            auto ptr = CreateBuffer();
            cudaMalloc(&ptr->m_pDeviceBuffer, size);
            return ptr;
        }

        [[nodiscard]] void* GetDeviceBuffer() noexcept { return m_pDeviceBuffer; }
        [[nodiscard]] const void* GetDeviceBuffer() const noexcept { return m_pDeviceBuffer; }
    private:
        static std::shared_ptr<Buffer> CreateBuffer()
        {
            return std::shared_ptr<Buffer>(new Buffer(), [](Buffer* pBuffer){
                cudaFree(pBuffer->m_pDeviceBuffer);
                delete pBuffer;
            });
        }
    private:
        void *m_pDeviceBuffer;
    };
}

#endif //AXON_TYPES_CUH
