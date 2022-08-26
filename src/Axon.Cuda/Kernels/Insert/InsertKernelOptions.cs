using Axon.Common.LinearAlgebra;
using Axon.Cuda.Common;
using Axon.Cuda.Common.Buffers;
using Axon.Cuda.Common.Execution;

namespace Axon.Cuda.Kernels;

public class InsertKernelOptions : KernelOptionsBase
{
    public GlobalMemoryBuffer Input { get; set; }
    public int Rows { get; set; }
    public int Columns { get; set; }
    public float Value { get; set; }

    public InsertKernelOptions(MatrixStorage input, MatrixStorage output, float value)
    {
        Input = input.Buffer as GlobalMemoryBuffer;
        Output = output.Buffer as GlobalMemoryBuffer;

        Rows = input.Rows;
        Columns = input.Columns;
        Value = value;
    }
}