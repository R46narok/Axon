using Axon.Common.LinearAlgebra;
using Axon.Cuda.Buffers;
using Axon.Cuda.Common;
using Axon.Cuda.Common.Execution;

namespace Axon.Cuda.Kernels;

public class RemoveKernelOptions : KernelOptionsBase
{
    public GpuBuffer Input { get; set; }
    public int Index { get; set; }
    public int Rows { get; set; }
    public int Columns { get; set; }

    public RemoveKernelOptions(MatrixStorage input, MatrixStorage output, int index)
    {
        Input = input.Buffer as GpuBuffer;
        Output = output.Buffer as GpuBuffer;

        Rows = input.Rows;
        Columns = input.Columns;
        Index = index;
    }
}