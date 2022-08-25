using Axon.Cuda.Buffers;

namespace Axon.Cuda.Common.Execution;

public class KernelOptionsBase
{
    public GpuBuffer Output { get; set; } 
}