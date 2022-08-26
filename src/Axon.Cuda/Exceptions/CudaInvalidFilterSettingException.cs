using Axon.Cuda.Debug;

namespace Axon.Cuda.Exceptions;

public class CudaInvalidFilterSettingException : CudaExceptionBase
{
    public CudaInvalidFilterSettingException(string message, string file, int line)
        : base(CudaErrorCode.InvalidFilterSetting, message, file, line)
    {

    }
}