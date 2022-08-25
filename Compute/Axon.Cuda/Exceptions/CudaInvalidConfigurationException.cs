using Axon.Cuda.Common.Interop;

namespace Axon.Cuda.Exceptions;

public class CudaInvalidConfigurationException : CudaExceptionBase
{
    public CudaInvalidConfigurationException(string message, string file, int line)
        : base(CudaErrorCode.InvalidConfiguration, message, file, line)
    {

    }
}