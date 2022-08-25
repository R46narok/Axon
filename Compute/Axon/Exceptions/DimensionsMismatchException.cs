using System;
using Axon.Common.LinearAlgebra;

namespace Axon.Exceptions;

public class DimensionsMismatchException : ArgumentException
{
    public MatrixStorage FirstOperand { get; init; }
    public MatrixStorage SecondOperand { get; init; }

    public DimensionsMismatchException(MatrixStorage firstOperand, MatrixStorage secondOperand, string message)
        : base(message)
    {
        FirstOperand = firstOperand;
        SecondOperand = secondOperand;
    }

    public static void ThrowIfNotEqual(MatrixStorage firstOperand, MatrixStorage secondOperand)
    {
        if (firstOperand.Rows != secondOperand.Rows || firstOperand.Columns != secondOperand.Columns)
            throw new DimensionsMismatchException(firstOperand, secondOperand, 
                $"Dimensions not equal [{firstOperand.Rows}x{firstOperand.Columns}] and [{secondOperand.Rows}x{secondOperand.Columns}]");
    }

    public static void ThrowIfNotEqual(MatrixStorage firstOperand, MatrixStorage secondOperand, MatrixStorage output)
    {
        ThrowIfNotEqual(firstOperand, secondOperand);
        ThrowIfNotEqual(firstOperand, output);
    }
}