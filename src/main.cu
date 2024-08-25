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

    //std::string img_path256 = "C:/Users/ben/Desktop/optimizing-convolution/images/256.png";
    //std::string img_path512 = "C:/Users/ben/Desktop/optimizing-convolution/images/512.png";
    //std::string img_path1024 = "C:/Users/ben/Desktop/optimizing-convolution/images/1024.png";
    //std::string img_path2048 = "C:/Users/ben/Desktop/optimizing-convolution/images/2048.png";
    std::string img_path4096 = "C:/Users/ben/Desktop/optimizing-convolution/images/4096.png";
    //std::string img_path8192 = "C:/Users/ben/Desktop/optimizing-convolution/images/8192.png";

    std::vector<cv::Mat> input_images;
    std::vector<cv::Mat> gray_input_images;
    std::vector<cv::Mat> gray_input_images_f32;

    // gray_input_images.push_back(cv::imread(img_path256, cv::IMREAD_GRAYSCALE));
    //gray_input_images.push_back(cv::imread(img_path512, cv::IMREAD_GRAYSCALE));
    gray_input_images.push_back(cv::imread(img_path1024, cv::IMREAD_GRAYSCALE));
    // gray_input_images.push_back(cv::imread(img_path2048, cv::IMREAD_GRAYSCALE));
    //gray_input_images.push_back(cv::imread(img_path4096, cv::IMREAD_GRAYSCALE));
    //gray_input_images.push_back(cv::imread(img_path8192, cv::IMREAD_GRAYSCALE));

    //for (const cv::Mat& e : gray_input_images) {
    //    try
    //    {
    //        //cv::Mat float_image;
    //        //e.convertTo(float_image, CV_32FC1, 1.0 / 255.0);
    //        //gray_input_images_f32.push_back(float_image);
    //    }
    //    catch (const std::exception& e)
    //    {
    //        std::cout << e.what() << std::endl;
    //    }
    //}

    std::vector <cv::Mat> output1;

    for (cv::Mat& e : gray_input_images) {
        output1.push_back(e.clone());
    }

    for (int i = 0; i < gray_input_images.size(); i++) {

        gf_1d_gpu(&gray_input_images.at(0), &output1.at(i), GAUSSIAN_3x3_global);
        gf_1d_gpu(&gray_input_images.at(0), &output1.at(i), GAUSSIAN_3x3_load_balance2_global);
        gf_1d_gpu(&gray_input_images.at(0), &output1.at(i), GAUSSIAN_3x3_load_balance4_global);
        gf_1d_gpu(&gray_input_images.at(0), &output1.at(i), GAUSSIAN_3x3_load_balance8_global);
        gf_1d_gpu(&gray_input_images.at(0), &output1.at(i), GAUSSIAN_3x3_load_balance12_global);
        gf_1d_gpu(&gray_input_images.at(0), &output1.at(i), GAUSSIAN_3x3_load_balance16_global);

        gf_1d_gpu(&gray_input_images.at(0), &output1.at(i), GAUSSIAN_3x3_vectorized2_global);
        gf_1d_gpu(&gray_input_images.at(0), &output1.at(i), GAUSSIAN_3x3_vectorized4_global);
        gf_1d_gpu(&gray_input_images.at(0), &output1.at(i), GAUSSIAN_3x3_vectorized8_global);
        gf_1d_gpu(&gray_input_images.at(0), &output1.at(i), GAUSSIAN_3x3_vectorized12_global);
        gf_1d_gpu(&gray_input_images.at(0), &output1.at(i), GAUSSIAN_3x3_vectorized16_global);

        gf_1d_gpu(&gray_input_images.at(0), &output1.at(i), GAUSSIAN_3x3_constant);
        gf_1d_gpu(&gray_input_images.at(0), &output1.at(i), GAUSSIAN_3x3_load_balance2_constant);
        gf_1d_gpu(&gray_input_images.at(0), &output1.at(i), GAUSSIAN_3x3_load_balance4_constant);
        gf_1d_gpu(&gray_input_images.at(0), &output1.at(i), GAUSSIAN_3x3_load_balance8_constant);
        gf_1d_gpu(&gray_input_images.at(0), &output1.at(i), GAUSSIAN_3x3_load_balance12_constant);
        gf_1d_gpu(&gray_input_images.at(0), &output1.at(i), GAUSSIAN_3x3_load_balance16_constant);

        gf_1d_gpu(&gray_input_images.at(0), &output1.at(i), GAUSSIAN_3x3_vectorized2_constant);
        gf_1d_gpu(&gray_input_images.at(0), &output1.at(i), GAUSSIAN_3x3_vectorized4_constant);
        gf_1d_gpu(&gray_input_images.at(0), &output1.at(i), GAUSSIAN_3x3_vectorized8_constant);
        gf_1d_gpu(&gray_input_images.at(0), &output1.at(i), GAUSSIAN_3x3_vectorized12_constant);
        gf_1d_gpu(&gray_input_images.at(0), &output1.at(i), GAUSSIAN_3x3_vectorized16_constant);

        gf_1d_gpu(&gray_input_images.at(0), &output1.at(i), GAUSSIAN_3x3_local);
        gf_1d_gpu(&gray_input_images.at(0), &output1.at(i), GAUSSIAN_3x3_load_balance2_local);
        gf_1d_gpu(&gray_input_images.at(0), &output1.at(i), GAUSSIAN_3x3_load_balance4_local);
        gf_1d_gpu(&gray_input_images.at(0), &output1.at(i), GAUSSIAN_3x3_load_balance8_local);
        gf_1d_gpu(&gray_input_images.at(0), &output1.at(i), GAUSSIAN_3x3_load_balance12_local);
        gf_1d_gpu(&gray_input_images.at(0), &output1.at(i), GAUSSIAN_3x3_load_balance16_local);

        gf_1d_gpu(&gray_input_images.at(0), &output1.at(i), GAUSSIAN_3x3_vectorized2_local);
        gf_1d_gpu(&gray_input_images.at(0), &output1.at(i), GAUSSIAN_3x3_vectorized4_local);
        gf_1d_gpu(&gray_input_images.at(0), &output1.at(i), GAUSSIAN_3x3_vectorized8_local);
        gf_1d_gpu(&gray_input_images.at(0), &output1.at(i), GAUSSIAN_3x3_vectorized12_local);
        gf_1d_gpu(&gray_input_images.at(0), &output1.at(i), GAUSSIAN_3x3_vectorized16_local);

        gf_1d_gpu(&gray_input_images.at(0), &output1.at(i), GAUSSIAN_3x3_shared);
        gf_1d_gpu(&gray_input_images.at(0), &output1.at(i), GAUSSIAN_3x3_load_balance2_shared);
        gf_1d_gpu(&gray_input_images.at(0), &output1.at(i), GAUSSIAN_3x3_load_balance4_shared);
        gf_1d_gpu(&gray_input_images.at(0), &output1.at(i), GAUSSIAN_3x3_load_balance8_shared);
        gf_1d_gpu(&gray_input_images.at(0), &output1.at(i), GAUSSIAN_3x3_load_balance12_shared);
        gf_1d_gpu(&gray_input_images.at(0), &output1.at(i), GAUSSIAN_3x3_load_balance16_shared);

        gf_1d_gpu(&gray_input_images.at(0), &output1.at(i), GAUSSIAN_3x3_vectorized2_shared);
        gf_1d_gpu(&gray_input_images.at(0), &output1.at(i), GAUSSIAN_3x3_vectorized4_shared);
        gf_1d_gpu(&gray_input_images.at(0), &output1.at(i), GAUSSIAN_3x3_vectorized8_shared);
        gf_1d_gpu(&gray_input_images.at(0), &output1.at(i), GAUSSIAN_3x3_vectorized12_shared);
        gf_1d_gpu(&gray_input_images.at(0), &output1.at(i), GAUSSIAN_3x3_vectorized16_shared);
    }
    return 0;
}
