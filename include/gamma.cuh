#pragma once

#include "dependencies.cuh"

enum GAMMA {
	GAMMA_default,
	GAMMA_recompute,
	GAMMA_constant,
	GAMMA_shared,
	GAMMA_noLUT,
	GAMMA_loadBalance,
	GAMMA_LoadBalance_constant,
	GAMMA_LoadBalance_shared,
	GAMMA_LoadBalance_recompute,
	GAMMA_vectorized,
	GAMMA_vectorized_constant,
	GAMMA_vectorized_shared,
	GAMMA_vectorized_recompute,
};

float gc_3d_single_thread(cv::Mat input_img, cv::Mat* output_img, float gamma);
float gc_3d_openmp(cv::Mat inputImg, cv::Mat* outputImg, float gamma);
float gc_3d_cppthreads(cv::Mat inputImg, cv::Mat* outputImg, float gamma);
float gc_3d_opencv(cv::Mat input_img, cv::Mat* output_img, float gamma);

float gc_3d_gpu(cv::Mat* outputImg, float gamma, GAMMA ver);
__global__ void k_init_LUT(float gamma);
__global__ void k_3D_gc(unsigned char* input, int rows, int cols);
__global__ void k_3D_gc_constant(unsigned char* input, int rows, int cols);
__global__ void k_3D_gc_shared(unsigned char* input, int rows, int cols);
__global__ void k_3D_gc_recompute(unsigned char* input, int rows, int cols, float gamma);
__global__ void k_3D_gc_noLUT(unsigned char* input, int rows, int cols, float gamma);
__global__ void k_3D_gc_load_balance(unsigned char* input, int rows, int cols, int load);
__global__ void k_3D_gc_vectorized(unsigned char* input, int rows, int cols, int load);
__global__ void k_3D_gc_kernel_fusion(unsigned char* input, int rows, int cols, float gamma);