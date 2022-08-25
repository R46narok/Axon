using Axon.Cuda.Common.Interop;

namespace Axon.Cuda.Exceptions;

public class CudaLaunchFailureException : CudaExceptionBase
{
    public CudaLaunchFailureException(string message, string file, int line)
        : base(CudaErrorCode.LaunchFailure, message, file, line)
    {

    }
}
