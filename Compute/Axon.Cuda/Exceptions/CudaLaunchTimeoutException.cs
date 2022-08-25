using Axon.Cuda.Debug;

namespace Axon.Cuda.Exceptions;

public class CudaLaunchTimeoutException : CudaExceptionBase
{
    public CudaLaunchTimeoutException(string message, string file, int line)
        : base(CudaErrorCode.LaunchTimeout, message, file, line)
    {

    }
}
