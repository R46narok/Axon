using Axon.Common.LinearAlgebra;
using Axon.Cuda.Buffers;
using Axon.Cuda.Common;
using Axon.Cuda.Common.Execution;

namespace Axon.Cuda.Kernels;

public class PointwiseKernelOptions : KernelOptionsBase
{
    public GpuBuffer FirstOperand { get; set; }
    public GpuBuffer SecondOperand { get; set; }
    public float Scale { get; set; }

    public PointwiseKernelOptions(MatrixStorage first, MatrixStorage second, MatrixStorage output, float scale = 1.0f)
    {
        FirstOperand = first.Buffer as GpuBuffer;
        SecondOperand = second.Buffer as GpuBuffer;
        Output = output.Buffer as GpuBuffer;
        Scale = scale;
    }
}

public class PointwiseOperationKernelOptions : KernelOptionsBase
{
    public GpuBuffer Operand { get; set; }

    public PointwiseOperationKernelOptions(MatrixStorage input, MatrixStorage output)
    {
        Operand = input.Buffer as GpuBuffer;
        Output = output.Buffer as GpuBuffer;
    }
}