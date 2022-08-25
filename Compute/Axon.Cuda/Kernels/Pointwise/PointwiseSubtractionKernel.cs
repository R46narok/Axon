using System;
using System.Runtime.InteropServices;
using Axon.Cuda.Common;
using Axon.Cuda.Buffers;
using Axon.Cuda.Common.Execution;
using Axon.Cuda.Common.Interop;

namespace Axon.Cuda.Kernels;

public class PointwiseSubtractionKernel : KernelBase<PointwiseKernelOptions>
{
    [DllImport(Dll.Name, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pointwise_subtraction")]
    private static extern void PointwiseSubtraction(IntPtr first, IntPtr second, IntPtr output, int length);
    
    public override void Invoke(PointwiseKernelOptions options)
    {
        var first = options.FirstOperand.Ptr;
        var second = options.SecondOperand.Ptr;
        var output = options.Output.Ptr;
        var length = options.FirstOperand.ByteWidth / sizeof(float);

        PointwiseSubtraction(first, second, output, (int)length);
    }
}