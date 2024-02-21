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

	std::string img_path256 = "C:/Users/ben/Desktop/cuda-img-enhancement/images/256.png";
	std::string img_path512 = "C:/Users/ben/Desktop/cuda-img-enhancement/images/512.png";
	std::string img_path1024 = "C:/Users/ben/Desktop/cuda-img-enhancement/images/1024.png";
	std::string img_path2048 = "C:/Users/ben/Desktop/cuda-img-enhancement/images/2048.png";
	std::string img_path4096 = "C:/Users/ben/Desktop/cuda-img-enhancement/images/4096.png";

	std::vector <cv::Mat> input_images;
	std::vector <cv::Mat> gray_input_images;

	input_images.push_back(cv::imread(img_path256, cv::IMREAD_COLOR));
	input_images.push_back(cv::imread(img_path512, cv::IMREAD_COLOR));
	input_images.push_back(cv::imread(img_path1024, cv::IMREAD_COLOR));
	input_images.push_back(cv::imread(img_path2048, cv::IMREAD_COLOR));
	input_images.push_back(cv::imread(img_path4096, cv::IMREAD_COLOR));

	gray_input_images.push_back(cv::imread(img_path256, cv::IMREAD_GRAYSCALE));
	gray_input_images.push_back(cv::imread(img_path512, cv::IMREAD_GRAYSCALE));
	gray_input_images.push_back(cv::imread(img_path1024, cv::IMREAD_GRAYSCALE));
	gray_input_images.push_back(cv::imread(img_path2048, cv::IMREAD_GRAYSCALE));
	gray_input_images.push_back(cv::imread(img_path4096, cv::IMREAD_GRAYSCALE));

	std::vector <cv::Mat> output2;
	std::vector <cv::Mat> output3;
	std::vector <cv::Mat> output4;
	std::vector <cv::Mat> output5;

	std::vector <cv::Mat> output2g;
	std::vector <cv::Mat> output3g;
	std::vector <cv::Mat> output4g;
	std::vector <cv::Mat> output5g;

	for (int j = 0; j < 5; j++) {
		output2.push_back(input_images.at(j).clone());
		output3.push_back(input_images.at(j).clone());
		output4.push_back(input_images.at(j).clone());
		output5.push_back(input_images.at(j).clone());

		output2g.push_back(gray_input_images.at(j).clone());
		output3g.push_back(gray_input_images.at(j).clone());
		output4g.push_back(gray_input_images.at(j).clone());
		output5g.push_back(gray_input_images.at(j).clone());
	}

	float elapsed = 0.0f;
	float gamma = 0.3f;

	for (cv::Mat& e : output3) {
		elapsed += gf_3d_gpu(&e, GAUSSIAN_load_balance);
	}
	for (cv::Mat& e : output3) {
		elapsed += gf_3d_gpu(&e, GAUSSIAN_load_balance);
	}
	for (cv::Mat& e : output3) {
		elapsed += gf_3d_gpu(&e, GAUSSIAN_load_balance);
	}
	for (cv::Mat& e : output3) {
		elapsed += gf_3d_gpu(&e, GAUSSIAN_load_balance);
	}
	for (cv::Mat& e : output3) {
		elapsed += gf_3d_gpu(&e, GAUSSIAN_load_balance);
	}
	for (cv::Mat& e : output3) {
		elapsed += gf_3d_gpu(&e, GAUSSIAN_load_balance);
	}
	for (cv::Mat& e : output3) {
		elapsed += gf_3d_gpu(&e, GAUSSIAN_load_balance);
	}



	//for (cv::Mat& e : output2g) {
	//	elapsed += gf_1d_gpu(&e, GAUSSIAN_load_balance);
	//}

	//cv::imshow("output4", output4.at(2));
	cv::imshow("output2", output2.at(2));
	cv::imshow("output3", output3.at(2));
	cv::waitKey(0);
	return 0;
}
