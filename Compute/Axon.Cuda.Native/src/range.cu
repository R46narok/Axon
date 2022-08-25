//
// Created by Acer on 22.7.2022 г..
//

#include "range.cuh"
#include <nvtx3/nvToolsExt.h>

void range_push(char* pName)
{
    nvtxRangePush(pName);
}

void range_pop()
{
    nvtxRangePop();
}
