#include "Data/Dataset.cuh"

#include "Core/Matrix.cuh"
#include "Data/OneHot.cuh"

#include <cstdio>
#include <sstream>
#include <fstream>
#include <filesystem>
#include <utility>
#include <array>

namespace Axon
{
    Dataset::Dataset(const std::string& path, const std::initializer_list<CsvDescriptor> &descriptors)
        : m_FileName(path), m_Descriptors(descriptors)
    {
        for (const auto& desc : m_Descriptors)
            m_HostBuffers[desc.Name] = thrust::host_vector<float>(desc.Width * desc.Height);

        CalculateOffsets();
        ProcessFile();
    }

    void Dataset::CalculateOffsets()
    {
        uint32_t offset = 0;
        for (auto& desc : m_Descriptors)
        {
            desc.Offset = offset;
            offset += desc.Width;
        }
    }

    void Dataset::ProcessFile()
    {
        std::array<char, 16384> buffer{};
        FILE* pFile = fopen(m_FileName.c_str(), "r");
        fgets(buffer.data(), buffer.size(), pFile);

        uint32_t line = 0;
        while(fgets(buffer.data(), buffer.size(), pFile) != nullptr)
        {
            uint32_t idx = 0;
            const char* tok;
            for (tok = strtok(buffer.data(), ",");
                 tok && *tok;
                 tok = strtok(nullptr, ","))
            {
                auto& descriptor = m_Descriptors[GetDescriptorIdx(idx)];
                auto& host = m_HostBuffers[descriptor.Name];

                auto value = (float)atof(tok) / descriptor.Normalize;
                host.data()[line * descriptor.Width + (idx - descriptor.Offset)] = value;

                idx++;
            }
        }

        fclose(pFile);
    }

    int Dataset::GetDescriptorIdx(int idx) const
    {
        for (int i = 0; i < m_Descriptors.size(); ++i)
            if (m_Descriptors[i].Offset <= idx) return i;
        return -1;
    }

    void Dataset::Copy(const std::string &name, const Matrix &matrix, int classes)
    {
        auto& host = m_HostBuffers[name];

        if (classes == -1)
        {
            thrust::copy(host.begin(), host.end(), matrix.Begin());
            return;
        }

        thrust::host_vector<float> encoded(classes * host.size());;
        OneHot::Encode(host, encoded, classes);
        thrust::copy(encoded.begin(), encoded.end(), matrix.Begin());
    }
}