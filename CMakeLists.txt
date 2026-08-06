cmake_minimum_required(VERSION 3.24)
project(parallel_radiative_transfer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

set(CMAKE_CUDA_ARCHITECTURES native) 
find_package(CUDAToolkit REQUIRED)

add_library(cub::cub INTERFACE IMPORTED)
target_include_directories(cub::cub INTERFACE ${CUDAToolkit_INCLUDE_DIRS})

add_library(raytracer_core STATIC
	src/device/rt1.cu
	src/device/rt4.cu
	src/device/utils.cu
	src/host/halo.cpp
)
target_include_directories(raytracer_core PUBLIC include)
target_link_libraries(raytracer_core 
	PUBLIC 
		CUDA::cudart
		cub::cub
)
set_target_properties(raytracer_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)
add_executable(raytracer_main src/host/main.cpp)
target_link_libraries(raytracer_main PRIVATE raytracer_core)
enable_testing()
add_subdirectory(tests)


