using Axon.Cuda.Debug;

namespace Axon.Cuda.Exceptions;

public class CudaMissingConfigurationException : CudaExceptionBase
{
    public CudaMissingConfigurationException(string message, string file, int line)
        : base(CudaErrorCode.MissingConfiguration, message, file, line)
    {

    }
}
