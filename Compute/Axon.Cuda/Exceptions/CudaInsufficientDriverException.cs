using Axon.Cuda.Common.Interop;

namespace Axon.Cuda.Exceptions;

public class CudaInsufficientDriverException : CudaExceptionBase
{
    public CudaInsufficientDriverException(string message, string file, int line)
        : base(CudaErrorCode.InsufficientDriver, message, file, line)
    {

    }
}