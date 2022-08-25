using Axon.Common.LinearAlgebra;
using Axon.Cuda.Buffers;
using Axon.Cuda.Common;
using Axon.Cuda.Common.Execution;

namespace Axon.Cuda.Kernels;

public class FunctionKernelOptions : KernelOptionsBase
{
    public GpuBuffer Input { get; set; }

    public FunctionKernelOptions(MatrixStorage input, MatrixStorage output)
    {
        Input = input.Buffer as GpuBuffer;
        Output = output.Buffer as GpuBuffer;
    }
}