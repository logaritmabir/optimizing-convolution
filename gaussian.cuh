#pragma once

#include "dependencies.cuh"

enum GAUSSIAN {
	GAUSSIAN_default,
	GAUSSIAN_unroll,
	GAUSSIAN_prefetch,
	GAUSSIAN_constant,
	GAUSSIAN_shared,
	GAUSSIAN_combined,
	GAUSSIAN_load_balance,
	GAUSSIAN_vectorized
};

float gf_1d_single_thread(cv::Mat input_img, cv::Mat* output_img);
float gf_1d_cppthreads(cv::Mat input_img, cv::Mat* output_img);
float gf_1d_openmp(cv::Mat input_img, cv::Mat* output_img);

float gf_3d_single_thread(cv::Mat input_img, cv::Mat* output_img);
float gf_3d_cppthreads(cv::Mat input_img, cv::Mat* output_img);
float gf_3d_openmp(cv::Mat input_img, cv::Mat* output_img);

float gf_1d_gpu(cv::Mat* output_img, GAUSSIAN ver);
__global__ void k_1D_gf(unsigned char* input, int rows, int cols, int mask_dim);
__global__ void k_1D_gf_shared(unsigned char* input, int rows, int cols, int mask_dim);
__global__ void k_1D_gf_constant(unsigned char* input, int rows, int cols, int mask_dim);
__global__ void k_1D_gf_combined(unsigned char* input, int rows, int cols, int mask_dim);
__global__ void k_1D_gf_unroll(unsigned char* input, int rows, int cols, int mask_dim);
__global__ void k_1D_gf_prefetch(unsigned char* input, int rows, int cols, int mask_dim);

float gf_3d_gpu(cv::Mat* output_img, GAUSSIAN ver);
__global__ void k_3D_gf(unsigned char* input, int rows, int cols, int mask_dim);
__global__ void k_3D_gf_shared(unsigned char* input, int rows, int cols, int mask_dim);
__global__ void k_3D_gf_constant(unsigned char* input, int rows, int cols, int mask_dim);
__global__ void k_3D_gf_combined(unsigned char* input, int rows, int cols, int mask_dim);
__global__ void k_3D_gf_prefetch(unsigned char* input, int rows, int cols, int mask_dim);
__global__ void k_3D_gf_unroll(unsigned char* input, int rows, int cols, int mask_dim);