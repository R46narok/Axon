using Axon.Cuda.Common.Interop;

namespace Axon.Cuda.Exceptions;

public class CudaInvalidTextureBindingException : CudaExceptionBase
{
    public CudaInvalidTextureBindingException(string message, string file, int line)
        : base(CudaErrorCode.InvalidTextureBinding, message, file, line)
    {

    }
}