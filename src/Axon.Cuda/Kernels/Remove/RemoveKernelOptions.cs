using Axon.Common.LinearAlgebra;
using Axon.Cuda.Common;
using Axon.Cuda.Common.Buffers;
using Axon.Cuda.Common.Execution;

namespace Axon.Cuda.Kernels;

public class RemoveKernelOptions : KernelOptionsBase
{
    public GlobalMemoryBuffer Input { get; set; }
    public int Index { get; set; }
    public int Rows { get; set; }
    public int Columns { get; set; }

    public RemoveKernelOptions(MatrixStorage input, MatrixStorage output, int index)
    {
        Input = input.Buffer as GlobalMemoryBuffer;
        Output = output.Buffer as GlobalMemoryBuffer;

        Rows = input.Rows;
        Columns = input.Columns;
        Index = index;
    }
}