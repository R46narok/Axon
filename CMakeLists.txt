cmake_minimum_required(VERSION 3.21)
project(Axon CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_library(Axon SHARED src/Core/Matrix.cu src/Core/Matrix.cuh src/Core/Library.cuh src/Kernels/PointwiseKernels.cu src/Kernels/PointwiseKernels.cuh src/Kernels/TransformationKernels.cu src/Kernels/TransformationKernels.cuh src/Kernels/ActivationKernels.cu src/Kernels/ActivationKernels.cuh src/Core/Activation.cu src/Core/Activation.cuh src/Neural/NeuralNetwork.cu src/Neural/NeuralNetwork.cuh src/Core/Random.cu src/Core/Random.cuh src/Data/Dataset.cu src/Data/Dataset.cuh src/Data/OneHot.cu src/Data/OneHot.cuh)
target_include_directories(Axon PUBLIC src/)
target_compile_definitions(Axon PRIVATE AXON_CORE _CRT_SECURE_NO_WARNINGS)

set_target_properties(Axon PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_executable(Axon.Demo src/main.cu)
target_link_libraries(Axon.Demo PRIVATE Axon)