using System;
using System.Runtime.InteropServices;
using Axon.Cuda.Common;
using Axon.Cuda.Common.Execution;

namespace Axon.Cuda.Kernels;

public class PointwiseAdditionKernel : KernelBase<PointwiseKernelOptions>
{
    [DllImport(Dll.Name, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pointwise_addition")]
    private static extern void PointwiseAddition(IntPtr first, IntPtr second, IntPtr output, int length);

    public override void Invoke(PointwiseKernelOptions options)
    {
        var first = options.FirstOperand.Ptr;
        var second = options.SecondOperand.Ptr;
        var output = options.Output.Ptr;
        var length = options.FirstOperand.ByteWidth / sizeof(float);
        
        PointwiseAddition(first, second, output, (int)length);
    }
}