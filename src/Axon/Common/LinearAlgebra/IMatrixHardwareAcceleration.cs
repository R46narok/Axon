using Axon.Common.Interfaces;

namespace Axon.Common.LinearAlgebra;

public interface IMatrixHardwareAcceleration
{
   public IRange GetRange();
   
   public MatrixStorage Add(MatrixStorage first, MatrixStorage second);
   public void Add(MatrixStorage first, MatrixStorage second, MatrixStorage output);

   public MatrixStorage Subtract(MatrixStorage first, MatrixStorage second);
   public void Subtract(MatrixStorage first, MatrixStorage second, MatrixStorage output);

   public MatrixStorage PointwiseMultiply(MatrixStorage first, MatrixStorage second);
   public void PointwiseMultiply(MatrixStorage first, MatrixStorage second, MatrixStorage output);

   public MatrixStorage Transpose(MatrixStorage matrix);
   public void Transpose(MatrixStorage matrix, MatrixStorage output);
   
   public MatrixStorage Multiply(MatrixStorage first, MatrixStorage second);
   public void Multiply(MatrixStorage first, MatrixStorage second, MatrixStorage output);

   public void PointwiseLog(MatrixStorage matrix, MatrixStorage output);
   public MatrixStorage PointwiseLog(MatrixStorage matrix);

   public MatrixStorage Add(MatrixStorage input, float scalar);
   public void Add(MatrixStorage input, MatrixStorage output, float scalar);

   public float Sum(MatrixStorage matrix);

   public MatrixStorage Subtract(MatrixStorage first, MatrixStorage second, float scale);
   public void Subtract(MatrixStorage first, MatrixStorage second, MatrixStorage output, float scale);
   
   public MatrixStorage Multiply(float scalar);
   public void Multiply(float scalar, MatrixStorage input, MatrixStorage output);
   
   public void ApplySigmoid(MatrixStorage matrix, MatrixStorage output);
   public void ApplySigmoidGradient(MatrixStorage matrix, MatrixStorage output);

   public void InsertColumn(MatrixStorage matrix, MatrixStorage output, float value);
   public MatrixStorage InsertColumn(MatrixStorage matrix, float value);
   
   public void InsertRow(MatrixStorage matrix, MatrixStorage output, float value);
   public MatrixStorage InsertRow(MatrixStorage matrix, float value);

   public void RemoveColumn(MatrixStorage matrix, MatrixStorage output);
   public MatrixStorage RemoveColumn(MatrixStorage matrix);
}