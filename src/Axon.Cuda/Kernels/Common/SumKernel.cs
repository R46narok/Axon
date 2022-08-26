using System;
using System.Runtime.InteropServices;
using Axon.Common.LinearAlgebra;
using Axon.Cuda.Common;
using Axon.Cuda.Common.Buffers;
using Axon.Cuda.Common.Execution;

namespace Axon.Cuda.Kernels;

public class SumKernelOptions : KernelOptionsBase
{
    public GlobalMemoryBuffer Input { get; set; }
    public int Rows { get; set; }
    public int Columns { get; set; }

    public SumKernelOptions(MatrixStorage input, GlobalMemoryBuffer output)
    {
        Input = input.Buffer as GlobalMemoryBuffer;
        Rows = input.Rows;
        Columns = input.Columns;
        Output = output;
    }
}

public class SumKernel : KernelBase<SumKernelOptions>
{
    [DllImport(Dll.Name, CallingConvention = CallingConvention.Cdecl, EntryPoint = "sum")]
    private static extern void Sum(IntPtr input, IntPtr output, int rows, int columns);

    public override void Invoke(SumKernelOptions options)
    {
        var input = options.Input.Ptr;
        var output = options.Output.Ptr;
        var rows = options.Rows;
        var columns = options.Columns;
        
        Sum(input, output, rows, columns);
    }
}
