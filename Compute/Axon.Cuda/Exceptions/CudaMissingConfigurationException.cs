using Axon.Cuda.Common.Interop;

namespace Axon.Cuda.Exceptions;

public class CudaMissingConfigurationException : CudaExceptionBase
{
    public CudaMissingConfigurationException(string message, string file, int line)
        : base(CudaErrorCode.MissingConfiguration, message, file, line)
    {

    }
}
