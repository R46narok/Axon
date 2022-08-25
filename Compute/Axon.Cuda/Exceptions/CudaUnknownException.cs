using Axon.Cuda.Common.Interop;

namespace Axon.Cuda.Exceptions;

public class CudaUnknownException : CudaExceptionBase
{
    public CudaUnknownException(string message, string file, int line)
        : base(CudaErrorCode.Unknown, message, file, line)
    {

    }
}