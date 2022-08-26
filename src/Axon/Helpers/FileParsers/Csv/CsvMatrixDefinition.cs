namespace Axon.Helpers.FileParsers.Csv;

public class CsvMatrixDefinition
{
    public int Rows { get; }
    public int Columns { get; }
    public int Offset { get; }
    public string Name { get; }
    public float Normalize { get; }

    public bool Label { get; set; }
    
    public CsvMatrixDefinition(int rows, int columns, int offset, string name, float normalize = 1.0f, bool label = false)
    {
        Rows = rows;
        Columns = columns;
        Offset = offset;
        Name = name;
        Normalize = normalize;
        Label = label;
    }
}