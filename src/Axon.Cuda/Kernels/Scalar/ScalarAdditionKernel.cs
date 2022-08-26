using System;
using System.Drawing;
using System.Runtime.InteropServices;
using Axon.Cuda.Common;
using Axon.Cuda.Common.Execution;

namespace Axon.Cuda.Kernels;

public class ScalarAdditionKernel : KernelBase<ScalarKernelOptions>
{
    [DllImport(Dll.Name, CallingConvention = CallingConvention.Cdecl, EntryPoint = "add_scalar")]
    private static extern void AddScalar(IntPtr input, IntPtr output, int length, float scalar);
    
    public override void Invoke(ScalarKernelOptions options)
    {
        var inputBuffer = options.Input.Ptr;
        var outputBuffer = options.Output.Ptr;
        var length = options.Output.ByteWidth / sizeof(float);
        var scalar = options.Scalar;

        AddScalar(inputBuffer, outputBuffer, (int)length, scalar);
    }
}