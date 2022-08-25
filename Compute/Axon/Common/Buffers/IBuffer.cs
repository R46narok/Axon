using System;

namespace Axon.Common.Buffers;

public interface IBuffer : IDisposable
{
    public int Offset { get; }
    public long ByteWidth { get; }

    public void Upload(float[] data);
    public float[]? Read();
    public void Reset();

    public IntPtr Ptr { get; }
}