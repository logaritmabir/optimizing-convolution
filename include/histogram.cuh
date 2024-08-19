#pragma once

#include "dependencies.cuh"

enum HISTOGRAM {
	HISTOGRAM_default,
	HISTOGRAM_shared,
	HISTOGRAM_reduce_branches,
	HISTOGRAM_recompute,
	HISTOGRAM_load_balance,
	HISTOGRAM_vectorized,
	HISTOGRAM_vectorized_shared,
	HISTOGRAM_vectorized_reduce_branches
};

float he_1d_single_thread(cv::Mat input_img, cv::Mat* output_img);
float he_1d_cppthreads(cv::Mat input_img, cv::Mat* output_img);
float he_1d_openmp(cv::Mat input_img, cv::Mat* output_img);

float he_3d_single_thread(cv::Mat input_img, cv::Mat* output_img);
float he_3d_cppthreads(cv::Mat input_img, cv::Mat* output_img);
float he_3d_openmp(cv::Mat input_img, cv::Mat* output_img);

float he_1d_gpu(cv::Mat* output_img, HISTOGRAM ver);
__global__ void k_1D_extract_histogram(unsigned char* input, int pixels);
__global__ void k_1D_normalize_cdf_equalization(int pixels);
__global__ void k_1D_equalize(unsigned char* input, int pixels);

__global__ void k_1D_extract_histogram_shared(unsigned char* input, int pixels);
__global__ void k_1D_normalize_cdf_equalization_shared(int pixels);
__global__ void k_1D_equalize_shared(unsigned char* input, int pixels);
__global__ void k_1D_equalize_shared_recompute(unsigned char* input, int pixels);

__global__ void k_1D_equalize_vectorized(unsigned char* input, int pixels, int load);


float he_3d_gpu(cv::Mat* output_img, HISTOGRAM ver);
__global__ void k_3D_extract_histogram(unsigned char* input, int pixels);
__global__ void k_3D_normalize_cdf_equalization(int pixels);
__global__ void k_3D_equalize(unsigned char* input, int pixels);

__global__ void k_3D_normalize_cdf_equalization_shared(int pixels);
__global__ void k_3D_extract_histogram_shared(unsigned char* input, int pixels);
__global__ void k_3D_equalize_shared(unsigned char* input, int pixels);

__global__ void k_3D_extract_histogram_rb(unsigned char* input, int total_channel_size);
__global__ void k_3D_extract_histogram_rb_shared(unsigned char* input, int total_channel_size);
__global__ void k_3D_normalize_cdf_equalization_rb(int pixels);
__global__ void k_3D_equalize_rb(unsigned char* input, int total_channel_size);
