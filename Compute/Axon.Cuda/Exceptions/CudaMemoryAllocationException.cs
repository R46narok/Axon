using Axon.Cuda.Common.Interop;

namespace Axon.Cuda.Exceptions;

public class CudaMemoryAllocationException : CudaExceptionBase
{
    public CudaMemoryAllocationException(string message, string file, int line)
        : base(CudaErrorCode.MemoryAllocation, message, file, line)
    {

    }
}