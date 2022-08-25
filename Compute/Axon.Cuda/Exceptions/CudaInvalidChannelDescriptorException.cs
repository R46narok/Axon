using Axon.Cuda.Common.Interop;

namespace Axon.Cuda.Exceptions;

public class CudaInvalidChannelDescriptorException : CudaExceptionBase
{
    public CudaInvalidChannelDescriptorException(string message, string file, int line)
        : base(CudaErrorCode.InvalidChannelDescriptor, message, file, line)
    {

    }
}