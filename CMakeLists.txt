cmake_minimum_required(VERSION 3.21)
project(axon CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_library(axon STATIC src/loss.cu src/network.cu src/layer.cu include/core/network.cuh include/core/mnist.cuh src/mnist.cu)

target_include_directories(axon PUBLIC include/)
target_compile_definitions(axon PRIVATE AXON_CORE _CRT_SECURE_NO_WARNINGS)
target_link_libraries(axon PUBLIC cudnn.lib cublas.lib)

set_target_properties(axon PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_executable(axon_mnist apps/main.cu)
target_link_libraries(axon_mnist PRIVATE axon)
