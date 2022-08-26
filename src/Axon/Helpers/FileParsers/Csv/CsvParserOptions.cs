namespace Axon.Helpers.FileParsers.Csv;

public class CsvParserOptions
{
    private int _offset = 0;
    
    internal List<CsvMatrixDefinition> Definitions { get; } = new();

    public void AddMatrixDefinition(string name, int rows, int columns, float normalize = 1.0f)
    {
        Definitions.Add(new CsvMatrixDefinition(rows, columns, _offset, name, normalize));
        _offset += columns;
    }
    public void AddLabelMatrixDefinition(string name, int rows, int columns)
    {
        Definitions.Add(new CsvMatrixDefinition(rows, columns, 1, name, 1.0f, true));
        _offset += 1;
    }
}