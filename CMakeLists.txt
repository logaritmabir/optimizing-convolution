cmake_minimum_required(VERSION 3.10)
project(optimizing-convolution LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 75)

enable_language(CUDA)

find_package(CUDA REQUIRED)

set(OpenCV_DIR C:/Users/ben/Desktop/opencv/build)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)

set(CUDA_FILES
    src/gaussian-kernels.cu
)

set(CUDA_HEADERS
    include/dependencies.cuh
    include/gaussian.cuh
)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -arch=sm_75 --expt-relaxed-constexpr -G -lineinfo")

add_executable(${PROJECT_NAME} ${CUDA_FILES} ${CUDA_HEADERS} src/main.cu)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${CUDA_LIBRARIES})
