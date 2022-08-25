using Axon.Cuda.Common.Interop;

namespace Axon.Cuda.Exceptions;

public class CudaInvalidFilterSettingException : CudaExceptionBase
{
    public CudaInvalidFilterSettingException(string message, string file, int line)
        : base(CudaErrorCode.InvalidFilterSetting, message, file, line)
    {

    }
}