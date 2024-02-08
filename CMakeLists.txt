cmake_minimum_required(VERSION 3.21)
project(cuda-demo)

enable_language(CUDA)

cmake_policy(SET CMP0104 NEW)
set(CMAKE_CUDA_ARCHITECTURES 75)

find_package(CUDAToolkit REQUIRED)

set(OpenCV_DIR "C:/Users/ben/Desktop/opencv/build")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(CUDA_FILES
    dependencies.cuh
    gamma-kernels.cu
    gamma.cuh
    gaussian-kernels.cu
    gaussian.cuh
    histogram-kernels.cu
    histogram.cuh
    multithreaded.cu
    singlethreaded.cu
)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=sm_75)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--expt-relaxed-constexpr)

add_executable(${PROJECT_NAME} ${CUDA_FILES} main.cu)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${CUDA_LIBRARIES})