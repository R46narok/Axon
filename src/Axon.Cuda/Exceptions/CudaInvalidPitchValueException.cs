using Axon.Cuda.Debug;

namespace Axon.Cuda.Exceptions;

public class CudaInvalidPitchValueException : CudaExceptionBase
{
    public CudaInvalidPitchValueException(string message, string file, int line)
        : base(CudaErrorCode.InvalidPitchValue, message, file, line)
    {

    }
}