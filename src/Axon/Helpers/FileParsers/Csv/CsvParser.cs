using Axon.Common.Buffers;
using Axon.Common.LinearAlgebra;
using Microsoft.VisualBasic.FileIO;

namespace Axon.Helpers.FileParsers.Csv;

public class CsvParser : IFileParser
{
   private readonly IBufferAllocator _allocator;

   public CsvParser(IBufferAllocator allocator)
   {
      _allocator = allocator;
   }

   public async Task<CsvParseResult> ParseAsync(string filename, Action<CsvParserOptions> optAction)
   {
      var options = new CsvParserOptions();
      optAction(options);
      var definitions = options.Definitions;
      
      using var stream = new StreamReader(filename);
      int line = 0;

      AllocateBuffers(out var buffers, definitions);
      
      await SkipHeaders(stream);
      while (!stream.EndOfStream)
      {
         var raw = (await stream.ReadLineAsync())!.Split(',');
         var fields = Array.ConvertAll(raw, s => float.Parse(s));

         ProcessFields(definitions, buffers, fields, line);

         ++line;
      }

      return CreateResultFromBuffers(buffers, definitions);
   }

   private async Task SkipHeaders(StreamReader stream)
   {
      await stream.ReadLineAsync();
   }

   private void ProcessFields(List<CsvMatrixDefinition> definitions, List<float[]> buffers, float[] fields, int line)
   {
      int definitionIdx = 0;
      for (int i = 0; i < fields.Length; ++i)
      {
         var definition = definitions[definitionIdx];
         
         if (definition.Label)
         {
            buffers[definitionIdx][(int)fields[i] * definition.Rows + line] = 1.0f;
         }
         else
         {
            buffers[definitionIdx][(i - definition.Offset) * definition.Rows + line] = fields[i] / definition.Normalize;
         }
         
         if (definition.Columns - 1 == i - definition.Offset || definition.Label) definitionIdx++;
      }
   }

   private void AllocateBuffers(out List<float[]> buffers, List<CsvMatrixDefinition> definitions)
   {
      buffers = new();
      foreach (var definition in definitions)
      {
         buffers.Add(new float[definition.Columns * definition.Rows]);
      }
   }

   private CsvParseResult CreateResultFromBuffers(List<float[]> buffers, List<CsvMatrixDefinition> definitions)
   {
      var result = new CsvParseResult();
      
      for (int i = 0; i < definitions.Count; ++i)
      {
         var definition = definitions[i];
         var buffer = buffers[i];

         var matrix = new MatrixStorage(definition.Rows, definition.Columns, _allocator);
         matrix.Buffer.Upload(buffer);
         result.Matrices.Add(definition.Name, matrix);
      }

      return result;
   }
}