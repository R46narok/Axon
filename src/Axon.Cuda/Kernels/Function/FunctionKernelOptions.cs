using Axon.Common.LinearAlgebra;
using Axon.Cuda.Common;
using Axon.Cuda.Common.Buffers;
using Axon.Cuda.Common.Execution;

namespace Axon.Cuda.Kernels;

public class FunctionKernelOptions : KernelOptionsBase
{
    public GlobalMemoryBuffer Input { get; set; }

    public FunctionKernelOptions(MatrixStorage input, MatrixStorage output)
    {
        Input = input.Buffer as GlobalMemoryBuffer;
        Output = output.Buffer as GlobalMemoryBuffer;
    }
}