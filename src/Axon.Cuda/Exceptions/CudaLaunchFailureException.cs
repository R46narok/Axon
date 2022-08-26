using Axon.Cuda.Debug;

namespace Axon.Cuda.Exceptions;

public class CudaLaunchFailureException : CudaExceptionBase
{
    public CudaLaunchFailureException(string message, string file, int line)
        : base(CudaErrorCode.LaunchFailure, message, file, line)
    {

    }
}
