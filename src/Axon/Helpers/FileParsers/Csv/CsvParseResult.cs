using Axon.Common.LinearAlgebra;

namespace Axon.Helpers.FileParsers.Csv;

public class CsvParseResult
{
    public Dictionary<string, MatrixStorage> Matrices { get; } = new ();

    public MatrixStorage this[string key] => Matrices[key];
}