using Axon.Common.LinearAlgebra;
using Axon.Cuda.Common;
using Axon.Cuda.Common.Buffers;
using Axon.Cuda.Common.Execution;

namespace Axon.Cuda.Kernels;

public class ScalarKernelOptions : KernelOptionsBase
{
    public GlobalMemoryBuffer Input { get; set; }
    public float Scalar { get; set; }

    public ScalarKernelOptions(MatrixStorage input, MatrixStorage output, float scalar)
    {
        Input = input.Buffer as GlobalMemoryBuffer;
        Output = output.Buffer as GlobalMemoryBuffer;
        Scalar = scalar;
    }
}