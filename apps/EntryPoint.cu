//
// Created by r46narok on 25.03.22 г..
//

#include <iostream>

#include "Axon/Types.cuh"
#include "Axon/Math/Vector.cuh"

using namespace Axon;

int main(int argc, char** ppArgv)
{
    Vector vec1 {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    Vector vec2 {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    std::cout << vec1.Transpose(vec2) << std::endl;
}
