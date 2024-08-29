#ifndef __CUDACC__  
#define __CUDACC__
#endif

#ifndef _DEPENDENCIES_
#define _DEPENDENCIES_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/opencv.hpp>

#define CHECK_CUDA_ERROR(val) check((val), __FILE__, __LINE__)
template <typename T>
void check(T err, const char* file, const int line)
{
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
		std::cerr << cudaGetErrorString(err) << std::endl;
		std::exit(EXIT_FAILURE);
	}
}

#endif 