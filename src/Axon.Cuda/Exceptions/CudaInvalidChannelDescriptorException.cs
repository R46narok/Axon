using Axon.Cuda.Debug;

namespace Axon.Cuda.Exceptions;

public class CudaInvalidChannelDescriptorException : CudaExceptionBase
{
    public CudaInvalidChannelDescriptorException(string message, string file, int line)
        : base(CudaErrorCode.InvalidChannelDescriptor, message, file, line)
    {

    }
}