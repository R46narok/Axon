using Axon.Common.Buffers;

namespace Axon.Cuda.Common.Buffers;

public class GlobalMemoryAllocator : IBufferAllocator
{
    public IBuffer Allocate(BufferDescriptor descriptor)
    {
        return new GlobalMemoryBuffer(descriptor);
    }

    public void Deallocate(IBuffer buffer)
    {
        buffer.Dispose();
    }

    public IBuffer TakeOwnership(IBuffer buffer, BufferDescriptor descriptor)
    {
        var cudaBuffer = buffer as GlobalMemoryBuffer;
        return new GlobalMemoryBuffer(cudaBuffer.Ptr + descriptor.Offset, descriptor);
    }

}