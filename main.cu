#ifndef __GPU__
#define __GPU__

#include "gaussian.cuh"
#include "gamma.cuh"
#include "histogram.cuh"

#endif

#include <string>
#include <cmath>
#include <chrono>
#include <math.h>

int main(int argc, char** argv)
{
	cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT); /*supressing annoying opencv infos in cmdline*/

	std::string img_path = "C:/Users/ben/Desktop/cuda-img-enhancement/images/4096.png";
	cv::Mat color_img = cv::imread(img_path, cv::IMREAD_COLOR);

	cv::Mat gray_img;
	cv::cvtColor(color_img, gray_img, cv::COLOR_BGR2GRAY);

	cv::Mat color_output = color_img.clone();
	cv::Mat gray_output = gray_img.clone();

	int iter = 5;
	float elapsed = 0.0f;
	float gamma = 0.3f;

	for (int i = 0; i < iter; i++) {
		elapsed += gc_3d_gpu(color_img, &color_output, gamma, GAMMA_default);
	}

	//cv::imshow("color-output", color_output);
	//cv::imshow("color-img", color_img);
	cv::waitKey(0);
	return 0;
}
