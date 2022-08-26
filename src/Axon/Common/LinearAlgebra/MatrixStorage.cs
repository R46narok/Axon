using System.Collections;
using Axon.Common.Buffers;

namespace Axon.Common.LinearAlgebra;

public class MatrixStorage : IEnumerable<float>
{

   public IBuffer Buffer { get; }

   public int Rows { get; private set; }
   public int Columns { get; private set; }
   public IBufferAllocator Allocator { get; }

   public MatrixStorage(int rows, int columns, IBufferAllocator allocator)
   {
      Rows = rows;
      Columns = columns;
      Allocator = allocator;

      Buffer = allocator.Allocate(new BufferDescriptor
      {
         ByteWidth = rows * columns * sizeof(float)
      });
   }

   public MatrixStorage(IBuffer buffer, int rows, int columns, IBufferAllocator allocator)
   {
      Rows = rows;
      Columns = columns;
      Buffer = buffer;
      Allocator = allocator;
   }

   public IEnumerator<float> GetEnumerator()
   {
      var data = Buffer.CopyToHost();
      foreach (var d in data)
      {
         yield return d;
      }
   }

   IEnumerator IEnumerable.GetEnumerator()
   {
      return GetEnumerator();
   }
}