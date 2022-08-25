using Axon.Cuda.Common.Interop;

namespace Axon.Cuda.Exceptions;

public class CudaInvalidDeviceFunctionException : CudaExceptionBase
{
    public CudaInvalidDeviceFunctionException(string message, string file, int line)
        : base(CudaErrorCode.InvalidDeviceFunction, message, file, line)
    {

    }
}