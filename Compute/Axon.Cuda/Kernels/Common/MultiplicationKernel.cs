using System;
using System.Runtime.InteropServices;
using Axon.Common.LinearAlgebra;
using Axon.Cuda.Common;
using Axon.Cuda.Common.Buffers;
using Axon.Cuda.Common.Execution;

namespace Axon.Cuda.Kernels;

public class MultiplicationKernelOptions : KernelOptionsBase
{
    public GlobalMemoryBuffer FirstOperand { get; set; }
    public GlobalMemoryBuffer SecondOperand { get; set; }

    public int FirstRows { get; set; }
    public int FirstColumns { get; set; }
    public int SecondColumns { get; set; }

    public MultiplicationKernelOptions()
    {
    }
    
    public MultiplicationKernelOptions(MatrixStorage first, MatrixStorage second, MatrixStorage output)
    {
        FirstOperand = first.Buffer as GlobalMemoryBuffer;
        SecondOperand = second.Buffer as GlobalMemoryBuffer;
        Output = output.Buffer as GlobalMemoryBuffer;

        FirstRows = first.Rows;
        FirstColumns = first.Columns;
        SecondColumns = second.Columns;
    }
}

public class MultiplicationKernel : KernelBase<MultiplicationKernelOptions>
{
    [DllImport(Dll.Name, CallingConvention = CallingConvention.Cdecl, EntryPoint = "multiply")]
    private static extern void Multiply(IntPtr first, IntPtr second, IntPtr output, 
        int firstRows, int firstColumns, int secondColumns);

    public override void Invoke(MultiplicationKernelOptions options)
    {
        var first = options.FirstOperand.Ptr;
        var second = options.SecondOperand.Ptr;
        var output = options.Output.Ptr;
        
        var firstRows = options.FirstRows;
        var firstColumns = options.FirstColumns;
        var secondColumns = options.SecondColumns;
        
        Multiply(first, second, output, firstRows, firstColumns, secondColumns);
    }
}
