using Axon.Cuda.Common.Interop;

namespace Axon.Cuda.Exceptions;

public class CudaInvalidResourceHandleException : CudaExceptionBase
{
    public CudaInvalidResourceHandleException(string message, string file, int line)
        : base(CudaErrorCode.InvalidResourceHandle, message, file, line)
    {

    }
}