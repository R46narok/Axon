using Axon.Cuda.Common.Interop;

namespace Axon.Cuda.Exceptions;

public class CudaInvalidDevicePointerException : CudaExceptionBase
{
    public CudaInvalidDevicePointerException(string message, string file, int line)
        : base(CudaErrorCode.InvalidDevicePointer, message, file, line)
    {

    }
}