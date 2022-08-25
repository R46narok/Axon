using System.Runtime.InteropServices;
using Axon.Common.Interfaces;
using Axon.Cuda.Common;

namespace Axon.Cuda.Debug;

public class NvtxRange : IRange
{
    [DllImport(Dll.Name, CallingConvention = CallingConvention.Cdecl, EntryPoint = "range_push")]
    public static extern void RangePush([MarshalAs(UnmanagedType.LPStr)] string name);
    
    [DllImport(Dll.Name, CallingConvention = CallingConvention.Cdecl, EntryPoint = "range_pop")]
    public static extern void RangePop();

    public void Push(string name) => RangePush(name);
    public void Pop() => RangePop();
}