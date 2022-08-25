using Axon.Cuda.Common.Interop;

namespace Axon.Cuda.Exceptions;

public class CudaSetOnActiveProcessException : CudaExceptionBase
{
    public CudaSetOnActiveProcessException(string message, string file, int line)
        : base(CudaErrorCode.SetOnActiveProcess, message, file, line)
    {

    }
}