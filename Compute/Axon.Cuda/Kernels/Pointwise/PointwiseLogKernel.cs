using System;
using System.Runtime.InteropServices;
using Axon.Cuda.Common;
using Axon.Cuda.Common.Execution;

namespace Axon.Cuda.Kernels;

public class PointwiseLogKernel : KernelBase<PointwiseOperationKernelOptions>
{
    [DllImport(Dll.Name, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pointwise_log")]
    private static extern void PointwiseLog(IntPtr input, IntPtr output, int length);
    
    public override void Invoke(PointwiseOperationKernelOptions options)
    {
        var input = options.Operand.Ptr;
        var output = options.Output.Ptr;
        var length = options.Output.ByteWidth / sizeof(float);
            
        PointwiseLog(input, output, (int)length);
    }
}
