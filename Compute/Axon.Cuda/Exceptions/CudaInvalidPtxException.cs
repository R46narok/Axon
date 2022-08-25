using Axon.Cuda.Common.Interop;

namespace Axon.Cuda.Exceptions;

public class CudaInvalidPtxException : CudaExceptionBase
{
    public CudaInvalidPtxException(string message, string file, int line)
        : base(CudaErrorCode.InvalidPtx, message, file, line)
    {

    }
}