using Axon.Cuda.Common.Interop;

namespace Axon.Cuda.Exceptions;

public class CudaStartupFailureException : CudaExceptionBase
{
    public CudaStartupFailureException(string message, string file, int line)
        : base(CudaErrorCode.StartupFailure, message, file, line)
    {

    }
}