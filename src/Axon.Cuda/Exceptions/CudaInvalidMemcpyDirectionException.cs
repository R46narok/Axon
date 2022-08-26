using Axon.Cuda.Debug;

namespace Axon.Cuda.Exceptions;

public class CudaInvalidMemcpyDirectionException : CudaExceptionBase
{
    public CudaInvalidMemcpyDirectionException(string message, string file, int line)
        : base(CudaErrorCode.InvalidMemcpyDirection, message, file, line)
    {

    }
}