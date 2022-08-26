using Axon.Cuda.Debug;

namespace Axon.Cuda.Exceptions;

public class CudaNoDeviceException : CudaExceptionBase
{
    public CudaNoDeviceException(string message, string file, int line)
        : base(CudaErrorCode.NoDevice, message, file, line)
    {

    }
}