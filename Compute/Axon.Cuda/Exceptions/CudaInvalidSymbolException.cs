using Axon.Cuda.Common.Interop;

namespace Axon.Cuda.Exceptions;

public class CudaInvalidSymbolException : CudaExceptionBase
{
    public CudaInvalidSymbolException(string message, string file, int line)
        : base(CudaErrorCode.InvalidSymbol, message, file, line)
    {

    }
}