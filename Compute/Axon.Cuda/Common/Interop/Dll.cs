using System.Runtime.InteropServices;

namespace Axon.Cuda.Common.Interop;

public static class Dll
{
   public const string Name = "Axon.Cuda.Native.Dll";
   public static CallingConvention Convention => CallingConvention.Cdecl;
}