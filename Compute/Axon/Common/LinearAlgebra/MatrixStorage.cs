using System;
using System.Collections;
using System.Collections.Generic;
using System.Threading.Tasks.Dataflow;
using Axon.Common.Buffers;
using Axon.Exceptions;

namespace Axon.Common.LinearAlgebra;

public class MatrixStorage : IEnumerable<float>
{
   public static IBufferAllocator BufferFactory { get; set; }
   public static IMatrixHardwareAcceleration Operations { get; set; }

   public IBuffer Buffer { get; }

   public int Rows { get; private set; }
   public int Columns { get; private set; }

   public MatrixStorage(int rows, int columns)
   {
      Rows = rows;
      Columns = columns;

      Buffer = BufferFactory!.Allocate(new BufferDescriptor
      {
         ByteWidth = rows * columns * sizeof(float)
      });
   }

   public MatrixStorage(IBuffer buffer, int rows, int columns)
   {
      Rows = rows;
      Columns = columns;
      Buffer = buffer;
   }

   public MatrixStorage Add(MatrixStorage other)
   {
      DimensionsMismatchException.ThrowIfNotEqual(this, other);
      return Operations.Add(this, other);
   }

   public void Add(MatrixStorage other, MatrixStorage output)
   {
      DimensionsMismatchException.ThrowIfNotEqual(this, other, output);
      Operations.Add(this, other, output);
   }

   public MatrixStorage Subtract(MatrixStorage other)
   {
      DimensionsMismatchException.ThrowIfNotEqual(this, other);
      return Operations.Subtract(this, other);
   }

   public void Subtract(MatrixStorage other, MatrixStorage output)
   {
      DimensionsMismatchException.ThrowIfNotEqual(this, other, output);
      Operations.Subtract(this, other, output);
   }

   public void Subtract(MatrixStorage other, MatrixStorage output, float scale)
   {
      DimensionsMismatchException.ThrowIfNotEqual(this, other, output);
      Operations.Subtract(this, other, output, scale);
   }

   public float Sum()
   {
      return Operations.Sum(this);
   }

   public MatrixStorage PointwiseMultiply(MatrixStorage other)
   {
      DimensionsMismatchException.ThrowIfNotEqual(this, other); 
      return Operations.PointwiseMultiply(this, other);
   }

   public void PointwiseMultiply(MatrixStorage other, MatrixStorage output)
   {
      DimensionsMismatchException.ThrowIfNotEqual(this, other, output);
      Operations.PointwiseMultiply(this, other, output);
   }

   public MatrixStorage Multiply(MatrixStorage other)
   {
      if (this.Columns != other.Rows)
             throw new ArgumentException();
      return Operations.Multiply(this, other);
   }

   public void Multiply(MatrixStorage other, MatrixStorage result)
   {
      if (this.Columns != other.Rows)
         throw new ArgumentException();
      Operations.Multiply(this, other, result);
   }

   public void Multiply(float scalar)
   {
      Operations.Multiply(scalar, this, this);
   }

   public void Multiply(float scalar, MatrixStorage output)
   {
      DimensionsMismatchException.ThrowIfNotEqual(this, output);
      Operations.Multiply(scalar, this, output);
   }

   public void PointwiseLog(MatrixStorage output)
   {
      DimensionsMismatchException.ThrowIfNotEqual(this, output);
      Operations.PointwiseLog(this, output);
   }

   public MatrixStorage PointwiseLog() => Operations.PointwiseLog(this);

   public MatrixStorage Add(float scalar) => Operations.Add(this, scalar);

   public void Add(float scalar, MatrixStorage output)
   {
      DimensionsMismatchException.ThrowIfNotEqual(this, output);
      Operations.Add(this, output, scalar);
   }

   public void ApplySigmoid(MatrixStorage output)
   {
      DimensionsMismatchException.ThrowIfNotEqual(this, output);
      Operations.ApplySigmoid(this, output);
   }

   public void ApplySigmoidGradient(MatrixStorage output)
   {
      DimensionsMismatchException.ThrowIfNotEqual(this, output);
      Operations.ApplySigmoidGradient(this, output);
   }

   public void InsertColumn(float value, MatrixStorage output)
   {
      if (this.Rows != output.Rows || this.Columns + 1 != output.Columns)
         throw new ArgumentException();
      Operations.InsertColumn(this, output, value);
   }

   public MatrixStorage InsertColumn(float value)
   {
      return Operations.InsertColumn(this, value);
   }

   public void InsertRow(float value, MatrixStorage output)
   {
      if (this.Rows != output.Rows + 1 || this.Columns != output.Columns)
         throw new ArgumentException();
      Operations.InsertRow(this, output, value);
   }

   public MatrixStorage InsertRow(float value) => Operations.InsertRow(this, value);

   public void RemoveColumn(MatrixStorage output)
   {
      if (this.Rows != output.Rows || this.Columns - 1 != output.Columns)
         throw new ArgumentException();
      Operations.RemoveColumn(this, output);
   }

   public MatrixStorage RemoveColumn() => Operations.RemoveColumn(this);

   public void Transpose(MatrixStorage output)
   {
      if (this.Rows != output.Columns || this.Columns != output.Rows)
         throw new ArgumentException();
      Operations.Transpose(this, output);
   }

   public MatrixStorage Transpose() => Operations.Transpose(this);
   public IEnumerator<float> GetEnumerator()
   {
      var data = Buffer.Read();
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