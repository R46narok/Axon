using Axon.Cuda.Common.Buffers;

namespace Axon.Cuda.Common.Execution;

public class KernelOptionsBase
{
    public GlobalMemoryBuffer Output { get; set; } 
}