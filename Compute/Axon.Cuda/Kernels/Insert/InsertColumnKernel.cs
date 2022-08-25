using System;
using System.Runtime.InteropServices;
using Axon.Cuda.Common;
using Axon.Cuda.Common.Execution;

namespace Axon.Cuda.Kernels;

public class InsertColumnKernel : KernelBase<InsertKernelOptions>
{
    [DllImport(Dll.Name, CallingConvention = CallingConvention.Cdecl, EntryPoint = "insert_column")]
    private static extern void InsertColumn(IntPtr input, IntPtr output, int rows, int columns, float value);

    public override void Invoke(InsertKernelOptions options)
    {
        var input = options.Input.Ptr;
        var output = options.Output.Ptr;
        var rows = options.Rows;
        var columns = options.Columns;
        var value = options.Value;
        
        InsertColumn(input, output, rows, columns, value);
    }
}
