using Axon.Cuda.Debug;

namespace Axon.Cuda.Exceptions;

public class CudaJitCompilationDisabledException : CudaExceptionBase
{
    public CudaJitCompilationDisabledException(string message, string file, int line)
        : base(CudaErrorCode.JitCompilationDisabled, message, file, line)
    {

    }
}