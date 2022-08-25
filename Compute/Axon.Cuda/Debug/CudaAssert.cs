using System.Runtime.InteropServices;
using Axon.Common.Interfaces;
using Axon.Cuda.Common;
using Axon.Cuda.Exceptions;

namespace Axon.Cuda.Debug;

public class CudaAssert : IAssert
{
    [DllImport(Dll.Name, CallingConvention = CallingConvention.Cdecl, EntryPoint = "set_error_callback")]
    private static extern void SetErrorCallback(IntPtr callback);
    
    private delegate void CallbackDelegate(int code, string message, string file, int line);

    private readonly CallbackDelegate _callback;
    
    public CudaAssert()
    {
        _callback = new CallbackDelegate(OnCudaError);
        SetErrorCallback(Marshal.GetFunctionPointerForDelegate(_callback));
    }

    private void OnCudaError(int code, string message, string file, int line)
    {
        Console.WriteLine($"{code} {message} {file} {line}");

        var cudaCode = (CudaErrorCode) code;
        switch (cudaCode)
        {
            case CudaErrorCode.Success: break;
            case CudaErrorCode.MissingConfiguration: throw new CudaMissingConfigurationException(message, file, line);
            case CudaErrorCode.MemoryAllocation: throw new CudaMemoryAllocationException(message, file, line);
            case CudaErrorCode.InitializationError: throw new CudaInitializationException(message, file, line);
            case CudaErrorCode.LaunchFailure: throw new CudaLaunchFailureException(message, file, line);
            case CudaErrorCode.LaunchTimeout: throw new CudaLaunchTimeoutException(message, file, line);
            case CudaErrorCode.LaunchOutOfResources: throw new CudaLaunchOutOfResourcesException(message, file, line);
            case CudaErrorCode.InvalidDeviceFunction: throw new CudaInvalidDeviceFunctionException(message, file, line);
            case CudaErrorCode.InvalidConfiguration: throw new CudaInvalidConfigurationException(message, file, line);
            case CudaErrorCode.InvalidDevice: throw new CudaInvalidDeviceException(message, file, line);
            case CudaErrorCode.InvalidValue: throw new CudaInvalidValueException(message, file, line);
            case CudaErrorCode.InvalidPitchValue: throw new CudaInvalidPitchValueException(message, file, line);
            case CudaErrorCode.InvalidSymbol: throw new CudaInvalidSymbolException(message, file, line);
            case CudaErrorCode.UnmapBufferObjectFailed: throw new CudaUnmapBufferObjectFailedException(message, file, line);
            case CudaErrorCode.InvalidDevicePointer: throw new CudaInvalidDevicePointerException(message, file, line);
            case CudaErrorCode.InvalidTexture: throw new CudaInvalidTextureException(message, file, line);
            case CudaErrorCode.InvalidTextureBinding: throw new CudaInvalidTextureBindingException(message, file, line);
            case CudaErrorCode.InvalidChannelDescriptor: throw new CudaInvalidChannelDescriptorException(message, file, line);
            case CudaErrorCode.InvalidMemcpyDirection: throw new CudaInvalidMemcpyDirectionException(message, file, line);
            case CudaErrorCode.InvalidFilterSetting: throw new CudaInvalidFilterSettingException(message, file, line);
            case CudaErrorCode.InvalidNormSetting: throw new CudaInvalidNormSettingException(message, file, line);
            case CudaErrorCode.Unknown: throw new CudaUnknownException(message, file, line);
            case CudaErrorCode.InvalidResourceHandle: throw new CudaInvalidResourceHandleException(message, file, line);
            case CudaErrorCode.InsufficientDriver: throw new CudaInsufficientDriverException(message, file, line);
            case CudaErrorCode.NoDevice: throw new CudaNoDeviceException(message, file, line);
            case CudaErrorCode.SetOnActiveProcess: throw new CudaSetOnActiveProcessException(message, file, line);
            case CudaErrorCode.StartupFailure: throw new CudaStartupFailureException(message, file, line);
            case CudaErrorCode.InvalidPtx: throw new CudaInvalidPtxException(message, file, line);
            case CudaErrorCode.UnsupportedPtxVersion: throw new CudaUnsupportedPtxVersionException(message, file, line);
            case CudaErrorCode.NoKernelImageForDevice: throw new CudaNoKernelImageForDeviceException(message, file, line);
            case CudaErrorCode.JitCompilerNotFound: throw new CudaJitCompilerNotFoundException(message, file, line);
            case CudaErrorCode.JitCompilationDisabled: throw new CudaJitCompilationDisabledException(message, file, line);
        }
    }
    
    public void Assert(int code)
    {
    }
}