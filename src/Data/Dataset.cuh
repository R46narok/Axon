#ifndef _AXON_CSV_PARSER_H
#define _AXON_CSV_PARSER_H

#include "Core/Library.cuh"

#include <string>
#include <map>
#include <thrust/host_vector.h>
#include <fstream>
#include <initializer_list>

namespace Axon
{
    class AXON_API Matrix;

    struct CsvDescriptor
    {
        std::string Name;
        uint32_t Width;
        uint32_t Height;
        float Normalize; // bool auto normalize
        uint32_t Offset;
    };

    class AXON_API Dataset
    {
    public:
        Dataset(const std::string& path, const std::initializer_list<CsvDescriptor>& descriptors);

        void Copy(const std::string& name, Matrix& matrix, int classes = -1);
    private:
        void ProcessFile();
        void CalculateOffsets();
        [[nodiscard]] int GetDescriptorIdx(int idx) const;
    private:
        std::vector<CsvDescriptor> m_Descriptors;
        std::map<std::string, thrust::host_vector<float>> m_HostBuffers;
        std::string m_FileName;
    };
}

#endif //_AXON_CSV_PARSER_H
