using Axon.Cuda.Common.Interop;

namespace Axon.Cuda.Exceptions;

public class CudaInvalidDeviceException : CudaExceptionBase
{
    public CudaInvalidDeviceException(string message, string file, int line)
        : base(CudaErrorCode.InvalidDevice, message, file, line)
    {

    }
}