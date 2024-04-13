#pragma once

#include "dependencies.cuh"

enum GAUSSIAN {
	GAUSSIAN_3x3_global,
	GAUSSIAN_3x3_local,
	GAUSSIAN_3x3_constant,
	GAUSSIAN_3x3_shared,

	GAUSSIAN_3x3_load_balance32_global,
	GAUSSIAN_3x3_load_balance16_global,
	GAUSSIAN_3x3_load_balance12_global,
	GAUSSIAN_3x3_load_balance8_global,
	GAUSSIAN_3x3_load_balance4_global,
	GAUSSIAN_3x3_load_balance2_global,		

	GAUSSIAN_3x3_vectorized32_global,
	GAUSSIAN_3x3_vectorized16_global,
	GAUSSIAN_3x3_vectorized12_global,
	GAUSSIAN_3x3_vectorized8_global,
	GAUSSIAN_3x3_vectorized4_global,
	GAUSSIAN_3x3_vectorized3_global,
	GAUSSIAN_3x3_vectorized2_global,

	GAUSSIAN_3x3_load_balance16_local,
	GAUSSIAN_3x3_load_balance12_local,
	GAUSSIAN_3x3_load_balance8_local,
	GAUSSIAN_3x3_load_balance4_local,
	GAUSSIAN_3x3_load_balance2_local,

	GAUSSIAN_3x3_vectorized16_local,
	GAUSSIAN_3x3_vectorized12_local,
	GAUSSIAN_3x3_vectorized8_local,
	GAUSSIAN_3x3_vectorized4_local,
	GAUSSIAN_3x3_vectorized2_local,

	GAUSSIAN_3x3_load_balance16_constant,
	GAUSSIAN_3x3_load_balance12_constant,
	GAUSSIAN_3x3_load_balance8_constant,
	GAUSSIAN_3x3_load_balance4_constant,
	GAUSSIAN_3x3_load_balance2_constant,

	GAUSSIAN_3x3_vectorized16_constant,
	GAUSSIAN_3x3_vectorized12_constant,
	GAUSSIAN_3x3_vectorized8_constant,
	GAUSSIAN_3x3_vectorized4_constant,
	GAUSSIAN_3x3_vectorized2_constant,

	GAUSSIAN_3x3_load_balance16_shared,
	GAUSSIAN_3x3_load_balance12_shared,
	GAUSSIAN_3x3_load_balance8_shared,
	GAUSSIAN_3x3_load_balance4_shared,
	GAUSSIAN_3x3_load_balance2_shared,

	GAUSSIAN_3x3_vectorized16_shared,
	GAUSSIAN_3x3_vectorized12_shared,
	GAUSSIAN_3x3_vectorized8_shared,
	GAUSSIAN_3x3_vectorized4_shared,
	GAUSSIAN_3x3_vectorized2_shared,
};

float gf_1d_single_thread(cv::Mat input_img, cv::Mat* output_img);
float gf_1d_cppthreads(cv::Mat input_img, cv::Mat* output_img);
float gf_1d_openmp(cv::Mat input_img, cv::Mat* output_img);

float gf_3d_single_thread(cv::Mat input_img, cv::Mat* output_img);
float gf_3d_cppthreads(cv::Mat input_img, cv::Mat* output_img);
float gf_3d_openmp(cv::Mat input_img, cv::Mat* output_img);

float gf_1d_gpu(cv::Mat* output_img, GAUSSIAN ver);

float gf_3d_gpu(cv::Mat* output_img, GAUSSIAN ver);
