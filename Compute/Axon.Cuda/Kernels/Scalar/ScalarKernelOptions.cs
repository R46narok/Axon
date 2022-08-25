using Axon.Common.LinearAlgebra;
using Axon.Cuda.Buffers;
using Axon.Cuda.Common;
using Axon.Cuda.Common.Execution;

namespace Axon.Cuda.Kernels;

public class ScalarKernelOptions : KernelOptionsBase
{
    public GpuBuffer Input { get; set; }
    public float Scalar { get; set; }

    public ScalarKernelOptions(MatrixStorage input, MatrixStorage output, float scalar)
    {
        Input = input.Buffer as GpuBuffer;
        Output = output.Buffer as GpuBuffer;
        Scalar = scalar;
    }
}