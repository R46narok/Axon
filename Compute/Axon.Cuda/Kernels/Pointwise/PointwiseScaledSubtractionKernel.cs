using System;
using System.Runtime.InteropServices;
using Axon.Cuda.Common;
using Axon.Cuda.Common.Execution;
using Axon.Cuda.Common.Interop;

namespace Axon.Cuda.Kernels;

public class PointwiseScaledSubtractionKernel : KernelBase<PointwiseKernelOptions>
{
    [DllImport(Dll.Name, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pointwise_scaled_subtraction")]
    private static extern void PointwiseScaledSubtraction(IntPtr first, IntPtr second, IntPtr output, int length, float scale);
    
    public override void Invoke(PointwiseKernelOptions options)
    {
        var first = options.FirstOperand.Ptr;
        var second = options.SecondOperand.Ptr;
        var output = options.Output.Ptr;
        var scale = options.Scale;
        var length = options.FirstOperand.ByteWidth / sizeof(float);
        
        PointwiseScaledSubtraction(first, second, output, (int)length, scale);
    }
}
