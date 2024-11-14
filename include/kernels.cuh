#pragma once

#include <stdint.h>
#include <stdio.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <npp.h>
#include <cudnn.h>

#include <arrayfire.h>

#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/opencv.hpp>

#define MAX_SM_SIZE 49152

#define IMTYPE_FLOAT

#if defined(IMTYPE_FLOAT)
    #define IMTYPE_SIZE 4
    typedef float imtype;
    typedef float4 imtype4;
    typedef float2 imtype2;
    #define make_imtype2 make_float2
    #define make_imtype4 make_float4
#elif defined(IMTYPE_UCHAR)
    #define IMTYPE_SIZE 1
    typedef unsigned char imtype;
    typedef uchar4 imtype4;
    typedef uchar2 imtype2;
    #define make_imtype2 make_uchar2
    #define make_imtype4 make_uchar4
#else
    #error "Invalid image type."
#endif

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

void launch_kernels(cv::Mat* input_img, cv::Mat* output_img);



