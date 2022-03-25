cmake_minimum_required(VERSION 3.21)
project(Axon CUDA)

set(CMAKE_CUDA_STANDARD 17)


add_subdirectory(src/)
add_subdirectory(apps/)



target_include_directories(${PROJECT_NAME} PUBLIC ./include)

target_link_libraries(${PROJECT_NAME}.App PRIVATE ${PROJECT_NAME})