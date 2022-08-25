using System;
using Axon.Common.Buffers;
using Axon.Common.Interfaces;
using Axon.Common.LinearAlgebra;
using Axon.Cuda.Common;
using Axon.Cuda.Common.Buffers;
using Axon.Cuda.Common.Execution;
using Axon.Cuda.Debug;
using Axon.Cuda.Kernels;

namespace Axon.Cuda.Operations;

public class CudaMatrixAcceleration : IMatrixHardwareAcceleration
{
    public IRange GetRange () => new NvtxRange();

    private readonly GlobalMemoryAllocator _allocator = new();
    
    public MatrixStorage Multiply(MatrixStorage first, MatrixStorage second)
    {
        var output = new MatrixStorage(first.Rows, second.Columns, _allocator);
        Multiply(first, second, output);
        return output;
    }

    public void Multiply(MatrixStorage first, MatrixStorage second, MatrixStorage output)
    {
        var options = new MultiplicationKernelOptions(first, second, output);
        InvokeKernel<MultiplicationKernel, MultiplicationKernelOptions>(options);
    }

    public MatrixStorage Add(MatrixStorage input, float scalar)
    {
        var output = new MatrixStorage(input.Rows, input.Columns, _allocator);
        Add(input, output, scalar);
        return output;
    }

    public void Add(MatrixStorage input, MatrixStorage output, float scalar)
    {
        var options = new ScalarKernelOptions(input, output, scalar);
        InvokeKernel<ScalarAdditionKernel, ScalarKernelOptions>(options);
    }

    public float Sum(MatrixStorage matrix)
    {
        var output = new GlobalMemoryBuffer(new BufferDescriptor {ByteWidth = sizeof(float)});
        var options = new SumKernelOptions(matrix, output);
        InvokeKernel<SumKernel, SumKernelOptions>(options);

        var cpuArray = new float[1];
        GlobalMemory.CopyDeviceToHost(output.Ptr, cpuArray, (int)output.ByteWidth);

        return cpuArray[0];
    }

    public MatrixStorage Multiply(float scalar)
    {
        throw new NotImplementedException();
    }

    public void Multiply(float scalar, MatrixStorage input, MatrixStorage output)
    {
        var options = new ScalarKernelOptions(input, output, scalar);
        InvokeKernel<ScalarMultiplicationKernel, ScalarKernelOptions>(options);
    }

    public MatrixStorage PointwiseLog(MatrixStorage matrix)
    {
        var output = new MatrixStorage(matrix.Rows, matrix.Columns, _allocator);
        PointwiseLog(matrix, output);
        return output;
    }
    
    public void PointwiseLog(MatrixStorage matrix, MatrixStorage output)
    {
        var options = new PointwiseOperationKernelOptions(matrix, output);
        InvokeKernel<PointwiseLogKernel, PointwiseOperationKernelOptions>(options);
    }
    
    public void ApplySigmoid(MatrixStorage matrix, MatrixStorage output)
    {
        var options = new FunctionKernelOptions(matrix, output);
        InvokeKernel<FunctionSigmoidKernel, FunctionKernelOptions>(options);
    }

    public void ApplySigmoidGradient(MatrixStorage matrix, MatrixStorage output)
    {
        var options = new FunctionKernelOptions(matrix, output);
        InvokeKernel<FunctionSigmoidGradientKernel, FunctionKernelOptions>(options);
    }

    public void InsertColumn(MatrixStorage matrix, MatrixStorage output, float value)
    {
        var options = new InsertKernelOptions(matrix, output, value);
        InvokeKernel<InsertColumnKernel, InsertKernelOptions>(options);
    }

    public MatrixStorage InsertColumn(MatrixStorage matrix, float value)
    {
        var output = new MatrixStorage(matrix.Rows, matrix.Columns + 1, _allocator);
        InsertColumn(matrix, output, value);
        return output;
    }
    
    public void RemoveColumn(MatrixStorage matrix, MatrixStorage output)
    {
        var options = new RemoveKernelOptions(matrix, output, 0);
        InvokeKernel<RemoveColumnKernel, RemoveKernelOptions>(options);
    }

    public MatrixStorage RemoveColumn(MatrixStorage matrix)
    {
        var output = new MatrixStorage(matrix.Rows, matrix.Columns - 1, _allocator);
        RemoveColumn(matrix, output);
        return output;
    }
    
    public void InsertRow(MatrixStorage matrix, MatrixStorage output, float value)
    {
        var options = new InsertKernelOptions(matrix, output, value);
        InvokeKernel<InsertRowKernel, InsertKernelOptions>(options);
    }

    public MatrixStorage InsertRow(MatrixStorage matrix, float value)
    {
        var output = new MatrixStorage(matrix.Rows + 1, matrix.Columns, _allocator);
        InsertRow(matrix, output, value);
        return output;
    }


    public MatrixStorage Add(MatrixStorage first, MatrixStorage second)
    {
        var output = new MatrixStorage(first.Rows, first.Columns, _allocator);
        Add(first, second, output);
        return output;
    }

    public void Add(MatrixStorage first, MatrixStorage second, MatrixStorage output)
    {
        var options = new PointwiseKernelOptions(first, second, output);
        InvokeKernel<PointwiseAdditionKernel, PointwiseKernelOptions>(options);
    }
    
    public MatrixStorage Subtract(MatrixStorage first, MatrixStorage second)
    {
        var output = new MatrixStorage(first.Rows, first.Columns, _allocator);
        Subtract(first, second, output);
        return output;
    }
    
    public void Subtract(MatrixStorage first, MatrixStorage second, MatrixStorage output)
    {
        var options = new PointwiseKernelOptions(first, second, output, first.Rows * first.Columns);
        InvokeKernel<PointwiseSubtractionKernel, PointwiseKernelOptions>(options);
    }

    public MatrixStorage Subtract(MatrixStorage first, MatrixStorage second, float scale)
    {
        var output = new MatrixStorage(first.Rows, first.Columns, _allocator);
        Subtract(first, second, output, scale);
        return output;
    }
    
    public void Subtract(MatrixStorage first, MatrixStorage second, MatrixStorage output, float scale)
    {
        var options = new PointwiseKernelOptions(first, second, output, scale);
        InvokeKernel<PointwiseScaledSubtractionKernel, PointwiseKernelOptions>(options);
    }
    
    public MatrixStorage PointwiseMultiply(MatrixStorage first, MatrixStorage second)
    {
        var output = new MatrixStorage(first.Rows, first.Columns, _allocator);
        PointwiseMultiply(first, second, output);
        return output;
    }
    
    public void PointwiseMultiply(MatrixStorage first, MatrixStorage second, MatrixStorage output)
    {
       var options = new PointwiseKernelOptions(first, second, output);
       InvokeKernel<PointwiseMultiplicationKernel, PointwiseKernelOptions>(options);
    }

    public MatrixStorage Transpose(MatrixStorage matrix)
    {
        var output = new MatrixStorage(matrix.Columns, matrix.Rows, _allocator);
        Transpose(matrix, output);
        return output;
    }

    public void Transpose(MatrixStorage matrix, MatrixStorage output)
    {
        var options = new TransposeKernelOptions(matrix, output);
        InvokeKernel<TransposeKernel, TransposeKernelOptions>(options);
    }

    private void InvokeKernel<TKernel, TOptions>(TOptions options)
        where TOptions : KernelOptionsBase
        where TKernel : KernelBase<TOptions>, new()
    {
        var kernel = new TKernel();
        kernel.Invoke(options);
    }
}