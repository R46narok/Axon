//
// Created by Acer on 17.10.2022 г..
//
#ifndef AXON_LIBRARY_CUH
#define AXON_LIBRARY_CUH

#ifdef AXON_CORE
#define AXON_API __declspec(dllexport)
#else
#define AXON_API __declspec(dllimport)
#endif

#include <memory>
#include <cstdint>

namespace Axon
{
    using precision_t = float;
}

#endif //AXON_LIBRARY_CUH
