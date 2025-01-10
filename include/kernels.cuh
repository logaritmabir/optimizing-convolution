#pragma once

#include <stdint.h>
#include <stdio.h>
#include <map>

#include <arrayfire.h>

#include "cuda-libs.cuh"
#include "opencv-libs.cuh"
#include "utils.cuh"

#define MAX_SM_SIZE 49152

#define IMTYPE_UCHAR

#if defined(IMTYPE_FLOAT)
    #define IMTYPE_SIZE 4
    #define IMAGE_TYPE CV_32FC1
    typedef float imtype;
    typedef float4 imtype4;
    typedef float2 imtype2;
    #define make_imtype2 make_float2
    #define make_imtype4 make_float4
#elif defined(IMTYPE_UCHAR)
    #define IMTYPE_SIZE 1
    #define IMAGE_TYPE CV_8UC1
    typedef unsigned char imtype;
    typedef uchar4 imtype4;
    typedef uchar2 imtype2;
    #define make_imtype2 make_uchar2
    #define make_imtype4 make_uchar4
#else
    #error "Invalid image type."
#endif

enum class KernelType {
    GM_3x3, CM_3x3, SM_3x3,
    GM_3x3_CF2, GM_3x3_CF4, GM_3x3_CF8, GM_3x3_CF12, GM_3x3_CF16,
    CM_3x3_CF2, CM_3x3_CF4, CM_3x3_CF8, CM_3x3_CF12, CM_3x3_CF16,
    SM_3x3_CF2, SM_3x3_CF4, SM_3x3_CF8, SM_3x3_CF12, SM_3x3_CF16,
    GM_3x3_CF2_Vec, GM_3x3_CF4_Vec, GM_3x3_CF8_Vec, GM_3x3_CF12_Vec, GM_3x3_CF16_Vec,
    CM_3x3_CF2_Vec, CM_3x3_CF4_Vec, CM_3x3_CF8_Vec, CM_3x3_CF12_Vec, CM_3x3_CF16_Vec,
    SM_3x3_CF2_Vec, SM_3x3_CF4_Vec, SM_3x3_CF8_Vec, SM_3x3_CF12_Vec, SM_3x3_CF16_Vec,
    ArrayFire, cuDNN, NPP, OpenCV
};

#define SM_REQUIRED_FOR_CF2 (34 * IMTYPE_SIZE * 66)
#define SM_REQUIRED_FOR_CF4 (34 * IMTYPE_SIZE * 130)
#define SM_REQUIRED_FOR_CF8 (34 * IMTYPE_SIZE * 258)
#define SM_REQUIRED_FOR_CF12 (34 * IMTYPE_SIZE * 386)
#define SM_REQUIRED_FOR_CF16 (34 * IMTYPE_SIZE * 514)

#if SM_REQUIRED_FOR_CF2 > MAX_SM_SIZE
    #define INSUFFICIENT_MEMORY_FOR_CF2
#endif

#if SM_REQUIRED_FOR_CF4 > MAX_SM_SIZE
    #define INSUFFICIENT_MEMORY_FOR_CF4
#endif

#if SM_REQUIRED_FOR_CF8 > MAX_SM_SIZE
    #define INSUFFICIENT_MEMORY_FOR_CF8
#endif

#if SM_REQUIRED_FOR_CF12 > MAX_SM_SIZE
    #define INSUFFICIENT_MEMORY_FOR_CF12
#endif

#if SM_REQUIRED_FOR_CF16 > MAX_SM_SIZE
    #define INSUFFICIENT_MEMORY_FOR_CF16
#endif

void launchKernels(cv::Mat* input_img, cv::Mat* output_img);
void testOutputs(cv::Mat* input_img, cv::Mat* output_img);
void callKernel(cv::Mat* input_img, cv::Mat* output_img, KernelType func);