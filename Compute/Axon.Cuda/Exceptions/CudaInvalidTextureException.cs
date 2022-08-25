using Axon.Cuda.Debug;

namespace Axon.Cuda.Exceptions;

public class CudaInvalidTextureException : CudaExceptionBase
{
    public CudaInvalidTextureException(string message, string file, int line)
        : base(CudaErrorCode.InvalidTexture, message, file, line)
    {

    }
}