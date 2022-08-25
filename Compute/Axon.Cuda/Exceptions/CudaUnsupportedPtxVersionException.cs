using Axon.Cuda.Common.Interop;

namespace Axon.Cuda.Exceptions;

public class CudaUnsupportedPtxVersionException : CudaExceptionBase
{
    public CudaUnsupportedPtxVersionException(string message, string file, int line)
        : base(CudaErrorCode.UnsupportedPtxVersion, message, file, line)
    {

    }
}