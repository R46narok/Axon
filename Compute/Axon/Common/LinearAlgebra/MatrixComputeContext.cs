using System;

namespace Axon.Common.LinearAlgebra;

public interface IMatrixComputationSelectionStage
{
    public IMatrixAdditionalSelectionStage PerformOn(MatrixStorage operand);
    public IMatrixOutput PerformOnSelf(MatrixStorage operand);

    public void PushRange(string name);
    public void PopRange();
}

public interface IMatrixAdditionalSelectionStage
{
    public IMatrixOutput And(MatrixStorage operand);
    public IMatrixOutput Into(MatrixStorage operand);
}

public interface IMatrixOutput
{
    public IMatrixComputationSelectionStage MultiplyInto(MatrixStorage output);
    public IMatrixComputationSelectionStage RemoveColumn();
    public IMatrixComputationSelectionStage InsertColumn(float scalar);
    public IMatrixComputationSelectionStage ApplySigmoidFunction();
    public IMatrixComputationSelectionStage ApplySigmoidGradientFunction();
    public IMatrixComputationSelectionStage MultiplyBy(float scalar);
    public IMatrixComputationSelectionStage Add(float scalar);
    public IMatrixComputationSelectionStage Log();
    public IMatrixComputationSelectionStage PointwiseMultiplyInto(MatrixStorage output);
    public IMatrixComputationSelectionStage PointwiseSubtractInto(MatrixStorage output);

    public IMatrixComputationSelectionStage Transpose();
}

public class MatrixComputeContext : IMatrixComputationSelectionStage, IMatrixAdditionalSelectionStage, IMatrixOutput
{
    private MatrixStorage? _firstOperand;
    private MatrixStorage? _secondOperand;
    private readonly IMatrixHardwareAcceleration _acceleration;
    
    private MatrixComputeContext(IMatrixHardwareAcceleration acceleration)
    {
        _acceleration = acceleration;
    }

    public static IMatrixComputationSelectionStage Create(IMatrixHardwareAcceleration operations) => new MatrixComputeContext(operations);

    public IMatrixAdditionalSelectionStage PerformOn(MatrixStorage operand)
    {
        _firstOperand = operand;
        return this;
    }

    public IMatrixOutput PerformOnSelf(MatrixStorage operand)
    {
        _firstOperand = operand;
        _secondOperand = operand;
        return this;
    }

    public void PushRange(string name)
    {
        _acceleration.GetRange().Push(name);
    }

    public void PopRange()
    {
        _acceleration.GetRange().Pop();
    }

    public IMatrixOutput And(MatrixStorage operand)
    {
        _secondOperand = operand;
        return this;
    }

    public IMatrixOutput Into(MatrixStorage operand)
    {
        _secondOperand = operand;
        return this;
    }

    public IMatrixComputationSelectionStage MultiplyInto(MatrixStorage output)
    {
        EnsureBothOperandsNotNull();
        _acceleration.Multiply(_firstOperand!, _secondOperand!, output);
        return ResetOperands();
    }

    public IMatrixComputationSelectionStage RemoveColumn()
    {
        EnsureBothOperandsNotNull();
        _acceleration.RemoveColumn(_firstOperand!, _secondOperand!);
        return ResetOperands();
    }

    public IMatrixComputationSelectionStage ApplySigmoidFunction()
    {
        EnsureBothOperandsNotNull();
        _acceleration.ApplySigmoid(_firstOperand!, _secondOperand!);
        return ResetOperands();
    }

    public IMatrixComputationSelectionStage ApplySigmoidGradientFunction()
    {
        EnsureBothOperandsNotNull();
        _acceleration.ApplySigmoidGradient(_firstOperand!, _secondOperand!);
        return ResetOperands();
    }

    public IMatrixComputationSelectionStage MultiplyBy(float scalar)
    {
        EnsureBothOperandsNotNull();
        _acceleration.Multiply(scalar, _firstOperand, _secondOperand);
        return ResetOperands();
    }

    public IMatrixComputationSelectionStage Add(float scalar)
    {
        EnsureBothOperandsNotNull();
        _acceleration.Add(_firstOperand, _secondOperand, scalar);
        return ResetOperands();
    }

    public IMatrixComputationSelectionStage Log()
    {
       EnsureBothOperandsNotNull();
       _acceleration.PointwiseLog(_firstOperand, _secondOperand);
       return ResetOperands();
    }

    public IMatrixComputationSelectionStage PointwiseMultiplyInto(MatrixStorage output)
    {
        EnsureBothOperandsNotNull();
        _acceleration.PointwiseMultiply(_firstOperand!, _secondOperand!, output);
        return ResetOperands();
    }

    public IMatrixComputationSelectionStage PointwiseSubtractInto(MatrixStorage output)
    {
        EnsureBothOperandsNotNull();
        _acceleration.Subtract(_firstOperand!, _secondOperand!, output);
        return ResetOperands();
    }

    public IMatrixComputationSelectionStage Transpose()
    {
        EnsureBothOperandsNotNull();
        _acceleration.Transpose(_firstOperand!, _secondOperand!);
        return ResetOperands();
    }

    public IMatrixComputationSelectionStage InsertColumn(float scalar)
    {
        EnsureBothOperandsNotNull();
        _acceleration.InsertColumn(_firstOperand!, _secondOperand!, scalar);
        return ResetOperands();
    }

    private void EnsureBothOperandsNotNull()
    {
        if (_firstOperand is null && _secondOperand is null)
            throw new ArgumentException();
    }

    private IMatrixComputationSelectionStage ResetOperands()
    {
        _firstOperand = null;
        _secondOperand = null;

        return this;
    }
}