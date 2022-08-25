using Axon.Common.Buffers;

namespace Axon.Cuda.Common.Buffers;

public class GlobalMemoryBuffer : IBuffer, IDisposable
{
    public int Offset { get; private set; }
    public long ByteWidth { get; private set; }
    public IntPtr Ptr { get; private set; }
    
    public GlobalMemoryBuffer(BufferDescriptor descriptor) 
    {
        Offset = descriptor.Offset;
        ByteWidth = descriptor.ByteWidth;
        if (descriptor.Offset == 0)
            Ptr = GlobalMemory.Malloc(ByteWidth);
    }

    public GlobalMemoryBuffer(IntPtr ptr, BufferDescriptor descriptor)
    {
        ByteWidth = descriptor.ByteWidth;
        Offset = descriptor.Offset;
        Ptr = ptr;
    }

    public void Upload(float[] data)
    {
        GlobalMemory.CopyHostToDevice(data, Ptr, ByteWidth);
    }

    public float[]? CopyToHost()
    {
        var cpuBuffer = new float[ByteWidth / sizeof(float)];
        GlobalMemory.CopyDeviceToHost(Ptr, cpuBuffer, ByteWidth);

        return cpuBuffer;
    }

    public void Reset()
    {
        GlobalMemory.Memset(Ptr, ByteWidth, 0);
    }

    public void Dispose()
    {
        GlobalMemory.Free(Ptr);
    }
}