cmake_minimum_required(VERSION 3.10)
project(optimizing-convolution LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 75)

enable_language(CUDA)

set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6")
set(ArrayFire_DIR "C:/Program Files/ArrayFire/v3/cmake")
set(OpenCV_DIR "C:/Users/steam/Desktop/opencv/install")

find_package(CUDAToolkit REQUIRED)
find_package(ArrayFire REQUIRED)
find_package(OpenCV REQUIRED)

if (NOT ArrayFire_FOUND)
    message(FATAL_ERROR "ArrayFire library not found!")
else()
    message(STATUS "ArrayFire found at: ${ArrayFire_DIR}")
    message(STATUS "ArrayFire Include Dirs: ${ArrayFire_INCLUDE_DIRS}")
    message(STATUS "ArrayFire Libraries: ${ArrayFire_LIBRARIES}")
endif()

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${ArrayFire_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)

set(CUDA_FILES
    src/main.cu
    src/kernels.cu
)

set(CUDA_HEADERS
    include/utils.cuh
    include/kernels.cuh
    include/opencv-libs.cuh
    include/cuda-libs.cuh
)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -arch=sm_75 --expt-relaxed-constexpr -lineinfo")

add_executable(${PROJECT_NAME}
    ${CUDA_FILES}
    ${CUDA_HEADERS}
)

target_link_libraries(
    ${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    CUDA::nppc
    CUDA::nppial
    CUDA::nppig
    CUDA::nppif
    ${ArrayFire_LIBRARIES}
)
