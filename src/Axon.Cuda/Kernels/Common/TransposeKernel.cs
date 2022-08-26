using System;
using System.Runtime.InteropServices;
using Axon.Common.LinearAlgebra;
using Axon.Cuda.Common;
using Axon.Cuda.Common.Buffers;
using Axon.Cuda.Common.Execution;

namespace Axon.Cuda.Kernels;

public class TransposeKernelOptions : KernelOptionsBase
{
    public GlobalMemoryBuffer Input { get; set; }
    public int Rows { get; set; }
    public int Columns { get; set; }

    public TransposeKernelOptions(MatrixStorage input, MatrixStorage output)
    {
        Input = input.Buffer as GlobalMemoryBuffer;
        Output = output.Buffer as GlobalMemoryBuffer;

        Rows = input.Rows;
        Columns = input.Columns;
    }
}

public class TransposeKernel : KernelBase<TransposeKernelOptions>
{
    [DllImport(Dll.Name, CallingConvention = CallingConvention.Cdecl, EntryPoint = "transpose")]
    private static extern void Transpose(IntPtr input, IntPtr output, int rows, int columns);

    public override void Invoke(TransposeKernelOptions options)
    {
        var input = options.Input.Ptr;
        var output = options.Output.Ptr;
        var rows = options.Rows;
        var columns = options.Columns;
        
        Transpose(input, output, rows, columns);
    }
}
