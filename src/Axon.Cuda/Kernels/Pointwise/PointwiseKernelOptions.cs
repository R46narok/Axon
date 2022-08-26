using Axon.Common.LinearAlgebra;
using Axon.Cuda.Common;
using Axon.Cuda.Common.Buffers;
using Axon.Cuda.Common.Execution;

namespace Axon.Cuda.Kernels;

public class PointwiseKernelOptions : KernelOptionsBase
{
    public GlobalMemoryBuffer FirstOperand { get; set; }
    public GlobalMemoryBuffer SecondOperand { get; set; }
    public float Scale { get; set; }

    public PointwiseKernelOptions(MatrixStorage first, MatrixStorage second, MatrixStorage output, float scale = 1.0f)
    {
        FirstOperand = first.Buffer as GlobalMemoryBuffer;
        SecondOperand = second.Buffer as GlobalMemoryBuffer;
        Output = output.Buffer as GlobalMemoryBuffer;
        Scale = scale;
    }
}

public class PointwiseOperationKernelOptions : KernelOptionsBase
{
    public GlobalMemoryBuffer Operand { get; set; }

    public PointwiseOperationKernelOptions(MatrixStorage input, MatrixStorage output)
    {
        Operand = input.Buffer as GlobalMemoryBuffer;
        Output = output.Buffer as GlobalMemoryBuffer;
    }
}