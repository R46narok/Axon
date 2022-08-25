using System;
using Axon.Common.Buffers;

namespace Axon.Cuda.Buffers;

public class GlobalMemoryAllocator : IBufferAllocator
{
    public IBuffer Allocate(BufferDescriptor descriptor)
    {
        return new GpuBuffer(descriptor);
    }

    public void Deallocate(IBuffer buffer)
    {
        buffer.Dispose();
    }

    public IBuffer TakeOwnership(IntPtr ptr, BufferDescriptor descriptor)
    {
        return new GpuBuffer(ptr + descriptor.Offset, descriptor);
    }

}