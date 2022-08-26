using Axon.Cuda.Debug;

namespace Axon.Cuda.Exceptions;

public class CudaUnmapBufferObjectFailedException : CudaExceptionBase
{
    public CudaUnmapBufferObjectFailedException(string message, string file, int line)
        : base(CudaErrorCode.UnmapBufferObjectFailed, message, file, line)
    {

    }
}