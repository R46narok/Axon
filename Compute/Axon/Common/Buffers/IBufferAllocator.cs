using System;

namespace Axon.Common.Buffers;

public interface IBufferAllocator
{
    IBuffer Allocate(BufferDescriptor descriptor);
    void Deallocate(IBuffer buffer);
    
    IBuffer TakeOwnership(IBuffer buffer, BufferDescriptor descriptor);
}