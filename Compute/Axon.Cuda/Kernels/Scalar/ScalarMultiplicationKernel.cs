using System;
using System.Runtime.InteropServices;
using Axon.Cuda.Buffers;
using Axon.Cuda.Common;
using Axon.Cuda.Common.Execution;
using Axon.Cuda.Common.Interop;

namespace Axon.Cuda.Kernels;


[KernelEntryPoint("multiply_scalar")]
public class ScalarMultiplicationKernel : KernelBase<ScalarKernelOptions>
{
    [DllImport(Dll.Name, CallingConvention = CallingConvention.Cdecl, EntryPoint = "multiply_scalar")]
    private static extern void MultiplyScalar(IntPtr input, IntPtr output, int length, float scalar);
    
    public override void Invoke(ScalarKernelOptions options)
    {
        var inputBuffer = options.Input.Ptr;
        var outputBuffer = options.Output.Ptr;
        var length = options.Output.ByteWidth / sizeof(float);
        var scalar = options.Scalar;

        MultiplyScalar(inputBuffer, outputBuffer, (int)length, scalar);
    }
}
