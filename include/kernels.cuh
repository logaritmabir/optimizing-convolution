#pragma once

typedef unsigned char imtype;

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

void launch_kernels(cv::Mat* input_img, cv::Mat* output_img);