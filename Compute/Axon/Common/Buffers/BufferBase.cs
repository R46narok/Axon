using System;

namespace Axon.Common.Buffers;

public class BufferBase : IBuffer
{
    public IntPtr Ptr { get; protected set; }
    public int Offset { get; protected set; }
    public long ByteWidth { get; protected set; }
    public virtual void Upload(float[] data)
    {
         
    }

    public virtual void Reset()
    {
         
    }

    public virtual float[]? Read()
    {
        return null;
    }
     
    protected BufferBase(BufferDescriptor descriptor)
    {
        ArgumentNullException.ThrowIfNull(descriptor);
        
        Offset = descriptor.Offset;
        ByteWidth = descriptor.ByteWidth;
    }

    public virtual void Dispose()
    {
    }
}