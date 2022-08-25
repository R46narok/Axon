using Axon.Cuda.Debug;

namespace Axon.Cuda.Exceptions;

public class CudaUnsupportedPtxVersionException : CudaExceptionBase
{
    public CudaUnsupportedPtxVersionException(string message, string file, int line)
        : base(CudaErrorCode.UnsupportedPtxVersion, message, file, line)
    {

    }
}