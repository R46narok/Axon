using Axon.Cuda.Common.Interop;

namespace Axon.Cuda.Exceptions;

public class CudaInvalidValueException : CudaExceptionBase
{
    public CudaInvalidValueException(string message, string file, int line)
        : base(CudaErrorCode.InvalidValue, message, file, line)
    {

    }
}