using Axon.Cuda.Debug;

namespace Axon.Cuda.Exceptions;

public class CudaJitCompilerNotFoundException : CudaExceptionBase
{
    public CudaJitCompilerNotFoundException(string message, string file, int line)
        : base(CudaErrorCode.JitCompilerNotFound, message, file, line)
    {

    }
}