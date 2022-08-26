using Axon.Cuda.Debug;

namespace Axon.Cuda.Exceptions;

public class CudaInitializationException : CudaExceptionBase
{
    public CudaInitializationException(string message, string file, int line)
        : base(CudaErrorCode.MemoryAllocation, message, file, line)
    {

    }
}
