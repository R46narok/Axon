using System;
using System.Runtime.InteropServices;
using Axon.Cuda.Common;
using Axon.Cuda.Common.Execution;

namespace Axon.Cuda.Kernels;

public class FunctionSigmoidGradientKernel : KernelBase<FunctionKernelOptions>
{
    [DllImport(Dll.Name, CallingConvention = CallingConvention.Cdecl, EntryPoint = "function_sigmoid_gradient")]
    private static extern void FunctionSigmoidGradient(IntPtr input, IntPtr output, int length);

    public override void Invoke(FunctionKernelOptions options)
    {
        var input = options.Input.Ptr;
        var output = options.Output.Ptr;
        var length = options.Output.ByteWidth / sizeof(float);
        
        FunctionSigmoidGradient(input, output, (int)length);
    }
}
