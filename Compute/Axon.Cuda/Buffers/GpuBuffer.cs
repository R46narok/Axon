using System;
using Axon.Common.Buffers;

namespace Axon.Cuda.Buffers;

public class GpuBuffer : BufferBase, IDisposable
{
    public GpuBuffer(BufferDescriptor descriptor) : base(descriptor)
    {
        if (descriptor.Offset == 0)
            Ptr = GlobalMemory.Malloc(ByteWidth);
    }

    public GpuBuffer(IntPtr ptr, BufferDescriptor descriptor) : base(descriptor)
    {
        Ptr = ptr;
    }
    
    public override void Upload(float[] data)
    {
        GlobalMemory.CopyHostToDevice(data, Ptr, ByteWidth);
    }

    public override float[]? Read()
    {
        var cpuBuffer = new float[ByteWidth / sizeof(float)];
        GlobalMemory.CopyDeviceToHost(Ptr, cpuBuffer, ByteWidth);

        return cpuBuffer;
    }

    public override void Reset()
    {
        GlobalMemory.Memset(Ptr, ByteWidth, 0);
    }

    public override void Dispose()
    {
        GlobalMemory.Free(Ptr);
    }
}