#ifndef __GPU__
#define __GPU__
#include "gaussian.cuh"
#endif

#include <string>
#include <cmath>
#include <chrono>
#include <math.h>

int main(int argc, char** argv)
{
    cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT); /*suppressing annoying OpenCV infos in cmdline*/
    std::string img_path4096 = "C:/Users/ben/Desktop/optimizing-convolution/images/4096.png";

    std::vector<cv::Mat> gray_input_images;
    
    gray_input_images.push_back(cv::imread(img_path4096, cv::IMREAD_GRAYSCALE));

    std::vector <cv::Mat> output1;

    for (cv::Mat& e : gray_input_images) {
        output1.push_back(e.clone());
    }

    gf_1d_gpu(&gray_input_images.at(0), &output1.at(0), GAUSSIAN_3x3_global);
    gf_1d_gpu(&gray_input_images.at(0), &output1.at(0), GAUSSIAN_3x3_constant);
    gf_1d_gpu(&gray_input_images.at(0), &output1.at(0), GAUSSIAN_3x3_shared);
    gf_1d_gpu(&gray_input_images.at(0), &output1.at(0), GAUSSIAN_3x3_load_balance2_global);
    gf_1d_gpu(&gray_input_images.at(0), &output1.at(0), GAUSSIAN_3x3_load_balance4_global);
    gf_1d_gpu(&gray_input_images.at(0), &output1.at(0), GAUSSIAN_3x3_load_balance8_global);
    gf_1d_gpu(&gray_input_images.at(0), &output1.at(0), GAUSSIAN_3x3_load_balance12_global);
    gf_1d_gpu(&gray_input_images.at(0), &output1.at(0), GAUSSIAN_3x3_load_balance16_global);
    gf_1d_gpu(&gray_input_images.at(0), &output1.at(0), GAUSSIAN_3x3_vectorized2_global);
    gf_1d_gpu(&gray_input_images.at(0), &output1.at(0), GAUSSIAN_3x3_vectorized4_global);
    gf_1d_gpu(&gray_input_images.at(0), &output1.at(0), GAUSSIAN_3x3_vectorized8_global);
    gf_1d_gpu(&gray_input_images.at(0), &output1.at(0), GAUSSIAN_3x3_vectorized12_global);
    gf_1d_gpu(&gray_input_images.at(0), &output1.at(0), GAUSSIAN_3x3_vectorized16_global);
    gf_1d_gpu(&gray_input_images.at(0), &output1.at(0), GAUSSIAN_3x3_load_balance2_constant);
    gf_1d_gpu(&gray_input_images.at(0), &output1.at(0), GAUSSIAN_3x3_load_balance4_constant);
    gf_1d_gpu(&gray_input_images.at(0), &output1.at(0), GAUSSIAN_3x3_load_balance8_constant);
    gf_1d_gpu(&gray_input_images.at(0), &output1.at(0), GAUSSIAN_3x3_load_balance12_constant);
    gf_1d_gpu(&gray_input_images.at(0), &output1.at(0), GAUSSIAN_3x3_load_balance16_constant);
    gf_1d_gpu(&gray_input_images.at(0), &output1.at(0), GAUSSIAN_3x3_vectorized2_constant);
    gf_1d_gpu(&gray_input_images.at(0), &output1.at(0), GAUSSIAN_3x3_vectorized4_constant);
    gf_1d_gpu(&gray_input_images.at(0), &output1.at(0), GAUSSIAN_3x3_vectorized8_constant);
    gf_1d_gpu(&gray_input_images.at(0), &output1.at(0), GAUSSIAN_3x3_vectorized12_constant);
    gf_1d_gpu(&gray_input_images.at(0), &output1.at(0), GAUSSIAN_3x3_vectorized16_constant);
    gf_1d_gpu(&gray_input_images.at(0), &output1.at(0), GAUSSIAN_3x3_load_balance2_shared);
    gf_1d_gpu(&gray_input_images.at(0), &output1.at(0), GAUSSIAN_3x3_load_balance4_shared);
    gf_1d_gpu(&gray_input_images.at(0), &output1.at(0), GAUSSIAN_3x3_load_balance8_shared);
    gf_1d_gpu(&gray_input_images.at(0), &output1.at(0), GAUSSIAN_3x3_load_balance12_shared);
    gf_1d_gpu(&gray_input_images.at(0), &output1.at(0), GAUSSIAN_3x3_load_balance16_shared);
    gf_1d_gpu(&gray_input_images.at(0), &output1.at(0), GAUSSIAN_3x3_vectorized2_shared);
    gf_1d_gpu(&gray_input_images.at(0), &output1.at(0), GAUSSIAN_3x3_vectorized4_shared);
    gf_1d_gpu(&gray_input_images.at(0), &output1.at(0), GAUSSIAN_3x3_vectorized8_shared);
    gf_1d_gpu(&gray_input_images.at(0), &output1.at(0), GAUSSIAN_3x3_vectorized12_shared);
    gf_1d_gpu(&gray_input_images.at(0), &output1.at(0), GAUSSIAN_3x3_vectorized16_shared);

    return 0;
}
