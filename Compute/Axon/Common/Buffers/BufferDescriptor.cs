namespace Axon.Common.Buffers;

public class BufferDescriptor
{
    public int Offset { get; set; }
    public long ByteWidth { get; set; }

    public BufferDescriptor(long byteWidth, int offset = 0)
    {
        ByteWidth = byteWidth;
        Offset = offset;
    }

    public BufferDescriptor() { }
}