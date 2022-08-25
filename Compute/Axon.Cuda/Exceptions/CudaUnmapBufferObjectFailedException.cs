using Axon.Cuda.Common.Interop;

namespace Axon.Cuda.Exceptions;

public class CudaUnmapBufferObjectFailedException : CudaExceptionBase
{
    public CudaUnmapBufferObjectFailedException(string message, string file, int line)
        : base(CudaErrorCode.UnmapBufferObjectFailed, message, file, line)
    {

    }
}