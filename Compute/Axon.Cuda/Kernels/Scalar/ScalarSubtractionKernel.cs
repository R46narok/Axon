using System;
using System.Runtime.InteropServices;
using Axon.Cuda.Common;
using Axon.Cuda.Common.Execution;
using Axon.Cuda.Common.Interop;

namespace Axon.Cuda.Kernels;

[KernelEntryPoint("subtract_scalar")]
public class ScalarSubtractionKernel : KernelBase<ScalarKernelOptions>
{
    [DllImport(Dll.Name, CallingConvention = CallingConvention.Cdecl, EntryPoint = "subtract_scalar")]
    private static extern void SubtractScalar(IntPtr input, IntPtr output, int length, float scalar);
    
    public override void Invoke(ScalarKernelOptions options)
    {
        var inputBuffer = options.Input.Ptr;
        var outputBuffer = options.Output.Ptr;
        var length = options.Output.ByteWidth / sizeof(float);
        var scalar = options.Scalar;

        SubtractScalar(inputBuffer, outputBuffer, (int)length, scalar);
    }
}
