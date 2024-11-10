#pragma once

#ifndef __CUDACC__  
#define __CUDACC__
#endif

#define CHECK_CUDA_ERROR(val) check_rt((val), __FILE__, __LINE__)
template <typename T>
void check_rt(T err, const char* file, const int line)
{
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
		std::cerr << cudaGetErrorString(err) << std::endl;
		std::exit(EXIT_FAILURE);
	}
}

#define CHECK_CUDNN_ERROR(val) check_dnn((val), __FILE__, __LINE__)
template <typename T>
void check_dnn(T stat, const char* file, const int line)
{
	if (stat != CUDNN_STATUS_SUCCESS)
	{
		std::cerr << "CUDNN Error at: " << file << ":" << line << std::endl;
		std::cerr << cudnnGetErrorString(stat) << std::endl;
		std::exit(EXIT_FAILURE);
	}
}

#define CHECK_ARRAYFIRE(val) check_dnn((val), __FILE__, __LINE__)
template <typename T>
void check_af(T err, const char* file, const int line)
{
	if (err != AF_SUCCESS)
	{
		std::cerr << "ArrayFire Error at: " << file << ":" << line << std::endl;
		std::cerr << af_err_to_string(err) << std::endl;
		std::exit(EXIT_FAILURE);
	}
}