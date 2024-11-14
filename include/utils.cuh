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

inline const char* getNppErrorString(NppStatus error) {
    switch (error) {
        case NPP_NOT_SUPPORTED_MODE_ERROR:           return "NPP_NOT_SUPPORTED_MODE_ERROR";
        case NPP_ROUND_MODE_NOT_SUPPORTED_ERROR:     return "NPP_ROUND_MODE_NOT_SUPPORTED_ERROR";
        case NPP_RESIZE_NO_OPERATION_ERROR:          return "NPP_RESIZE_NO_OPERATION_ERROR";
        case NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY:  return "NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY";
        default:                                     return "Unknown NPP error";
    }
}

#define CHECK_NPP_ERROR(val) check_npp((val), __FILE__, __LINE__)
template <typename T>
void check_npp(T err, const char* file, const int line)
{
	if (err != NPP_SUCCESS)
	{
		std::cerr << "Npp Error at: " << file << ":" << line << std::endl;
		std::cerr << getNppErrorString(err) << std::endl;
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

#define CHECK_ARRAYFIRE(val) check_af((val), __FILE__, __LINE__)
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