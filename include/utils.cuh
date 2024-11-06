#pragma once

#ifndef __CUDACC__  
#define __CUDACC__
#endif

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