using Axon.Cuda.Debug;

namespace Axon.Cuda.Exceptions;

public class CudaInvalidNormSettingException : CudaExceptionBase
{
    public CudaInvalidNormSettingException(string message, string file, int line)
        : base(CudaErrorCode.InvalidNormSetting, message, file, line)
    {

    }
}