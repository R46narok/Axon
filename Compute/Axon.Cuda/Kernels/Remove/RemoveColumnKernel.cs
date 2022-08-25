using System;
using System.Runtime.InteropServices;
using Axon.Cuda.Common;
using Axon.Cuda.Buffers;
using Axon.Cuda.Common.Execution;
using Axon.Cuda.Common.Interop;

namespace Axon.Cuda.Kernels;

[KernelEntryPoint("remove_column")]
public class RemoveColumnKernel : KernelBase<RemoveKernelOptions>
{
    [DllImport(Dll.Name, CallingConvention = CallingConvention.Cdecl, EntryPoint = "remove_column")]
    private static extern void RemoveColumn(IntPtr input, IntPtr output, int rows, int columns);

    public override void Invoke(RemoveKernelOptions options)
    {
        var input = options.Input.Ptr;
        var output = options.Output.Ptr;
        var rows = options.Rows;
        var columns = options.Columns;
        
        RemoveColumn(input, output, rows, columns);
    }
}