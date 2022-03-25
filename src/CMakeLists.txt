add_library(${PROJECT_NAME} SHARED Core.cu ../include/Axon/Types.cuh ../include/Axon/Math/Vector.cuh Vector.cu)
target_compile_definitions(${PROJECT_NAME} PRIVATE _AXON_CORE)
set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
