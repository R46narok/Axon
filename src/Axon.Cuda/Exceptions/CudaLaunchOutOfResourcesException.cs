using Axon.Cuda.Debug;

namespace Axon.Cuda.Exceptions;

public class CudaLaunchOutOfResourcesException : CudaExceptionBase
{
    public CudaLaunchOutOfResourcesException(string message, string file, int line)
        : base(CudaErrorCode.LaunchOutOfResources, message, file, line)
    {

    }
}
