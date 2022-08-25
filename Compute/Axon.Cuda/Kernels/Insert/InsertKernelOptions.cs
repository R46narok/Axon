using Axon.Common.LinearAlgebra;
using Axon.Cuda.Buffers;
using Axon.Cuda.Common;
using Axon.Cuda.Common.Execution;

namespace Axon.Cuda.Kernels;

public class InsertKernelOptions : KernelOptionsBase
{
    public GpuBuffer Input { get; set; }
    public int Rows { get; set; }
    public int Columns { get; set; }
    public float Value { get; set; }

    public InsertKernelOptions(MatrixStorage input, MatrixStorage output, float value)
    {
        Input = input.Buffer as GpuBuffer;
        Output = output.Buffer as GpuBuffer;

        Rows = input.Rows;
        Columns = input.Columns;
        Value = value;
    }
}