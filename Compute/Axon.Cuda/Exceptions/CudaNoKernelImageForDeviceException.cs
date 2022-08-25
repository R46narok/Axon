using Axon.Cuda.Common.Interop;

namespace Axon.Cuda.Exceptions;

public class CudaNoKernelImageForDeviceException : CudaExceptionBase
{
    public CudaNoKernelImageForDeviceException(string message, string file, int line)
        : base(CudaErrorCode.NoKernelImageForDevice, message, file, line)
    {

    }
}