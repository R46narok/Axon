using System;
using System.ComponentModel.DataAnnotations;

namespace Axon.Common.Buffers;

public enum BufferDataType : byte
{
    Unknown,
    Double,
    Float
}

public record class BufferBatchElement(int ByteWidth, BufferDataType DataType, string Name);

public class BufferBatch : IDisposable
{
    private readonly IBufferAllocator _allocator;
    private IBuffer[] _buffers = null!;
    private IBuffer _memoryPool = null!;
    
    public IBuffer this[int idx] => _buffers[idx];
    
    public BufferBatch(IBufferAllocator allocator, BufferBatchElement[] elements)
    {
        _allocator = allocator;
        InitializeMemoryPool(elements);
        InitializeBuffers(elements);
    }

    private void InitializeBuffers(BufferBatchElement[] elements)
    {
        int length = elements.Length;
        int offset = 0;
        _buffers = new IBuffer[length];
        
        for (int i = 0; i < length; ++i)
        {
            var element = elements[i];
            var descriptor = new BufferDescriptor(element.ByteWidth, offset);
            var buffer = _allocator.TakeOwnership(_memoryPool.Ptr, descriptor);

            _buffers[i] = buffer;
            offset += element.ByteWidth;
        }
    }

    private void InitializeMemoryPool(BufferBatchElement[] elements)
    {
        long byteWidth = 0;
        foreach (var element in elements)
        {
            byteWidth += element.ByteWidth;
        }

        var descriptor = new BufferDescriptor(byteWidth);
        _memoryPool = _allocator.Allocate(descriptor);
    }
    
    public void Dispose()
    {
        _memoryPool.Dispose();
    }
}