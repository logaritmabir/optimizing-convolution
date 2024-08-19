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
    cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT); /*suppressing annoying OpenCV infos in cmdline*/

    std::string img_path256 = "C:/Users/ben/Desktop/optimizing-convolution/images/256.png";
    std::string img_path512 = "C:/Users/ben/Desktop/optimizing-convolution/images/512.png";
    std::string img_path1024 = "C:/Users/ben/Desktop/optimizing-convolution/images/1024.png";
    std::string img_path2048 = "C:/Users/ben/Desktop/optimizing-convolution/images/2048.png";
    std::string img_path4096 = "C:/Users/ben/Desktop/optimizing-convolution/images/4096.png";
    std::string img_path8192 = "C:/Users/ben/Desktop/optimizing-convolution/images/8192.png";

    std::vector<cv::Mat> input_images;
    std::vector<cv::Mat> gray_input_images;
    std::vector<cv::Mat> gray_input_images_f32;

    gray_input_images.push_back(cv::imread(img_path256, cv::IMREAD_GRAYSCALE));
    gray_input_images.push_back(cv::imread(img_path512, cv::IMREAD_GRAYSCALE));
    gray_input_images.push_back(cv::imread(img_path1024, cv::IMREAD_GRAYSCALE));
    gray_input_images.push_back(cv::imread(img_path2048, cv::IMREAD_GRAYSCALE));
    gray_input_images.push_back(cv::imread(img_path4096, cv::IMREAD_GRAYSCALE));
    gray_input_images.push_back(cv::imread(img_path8192, cv::IMREAD_GRAYSCALE));

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

    float elapsed = 0.0f;

    std::vector <cv::Mat> output1;
    std::vector <cv::Mat> output2;
    std::vector <cv::Mat> output3;
    std::vector <cv::Mat> output4;
    std::vector <cv::Mat> output5;
    std::vector <cv::Mat> output6;
    std::vector <cv::Mat> output7;
    std::vector <cv::Mat> output8;

    for (cv::Mat& e: gray_input_images) {
        output1.push_back(e.clone());
        output2.push_back(e.clone());
        output3.push_back(e.clone());
        output4.push_back(e.clone());
        output5.push_back(e.clone());
        output6.push_back(e.clone());
        output7.push_back(e.clone());
        output8.push_back(e.clone());
    }

    for(int i = 0; i < gray_input_images.size(); i++ ){
        gf_1d_gpu(&gray_input_images.at(i), &output1.at(i), GAUSSIAN_3x3_global);
    }

     //for (cv::Mat& e : output1) {
     //    elapsed += gf_1d_gpu(&e, GAUSSIAN_3x3_load_balance2_global);
     //}

     //for (cv::Mat& e : output2) {
     //    elapsed += gf_1d_gpu(&e, GAUSSIAN_3x3_load_balance4_global);
     //}

     //for (cv::Mat& e : output3) {
     //    elapsed += gf_1d_gpu(&e, GAUSSIAN_3x3_load_balance8_global);
     //}

     //for (cv::Mat& e : output4) {
     //    elapsed += gf_1d_gpu(&e, GAUSSIAN_3x3_load_balance12_global);
     //}

     //for (cv::Mat& e : output5) {
     //   elapsed += gf_1d_gpu(&e, GAUSSIAN_3x3_load_balance16_global);
     //}



    // for (cv::Mat& e : output1) {
    //     elapsed += gf_1d_gpu(&e, GAUSSIAN_3x3_vectorized2_global);
    // }

    // for (cv::Mat& e : output2) {
    //     elapsed += gf_1d_gpu(&e, GAUSSIAN_3x3_vectorized4_global);
    // }

    // for (cv::Mat& e : output3) {
    //     elapsed += gf_1d_gpu(&e, GAUSSIAN_3x3_vectorized8_global);
    // }

    // for (cv::Mat& e : output4) {
    //     elapsed += gf_1d_gpu(&e, GAUSSIAN_3x3_vectorized12_global);
    // }

    // for (cv::Mat& e : output5) {
    //     elapsed += gf_1d_gpu(&e, GAUSSIAN_3x3_vectorized16_global);
    // }


    // for (cv::Mat& e : output1) {
    //     elapsed += gf_1d_gpu(&e, GAUSSIAN_3x3_local);
    // }

    // for (cv::Mat& e : output1) {
    //     elapsed += gf_1d_gpu(&e, GAUSSIAN_3x3_load_balance2_local);
    // }

    // for (cv::Mat& e : output2) {
    //     elapsed += gf_1d_gpu(&e, GAUSSIAN_3x3_load_balance4_local);
    // }

    // for (cv::Mat& e : output3) {
    //     elapsed += gf_1d_gpu(&e, GAUSSIAN_3x3_load_balance8_local);
    // }

    // for (cv::Mat& e : output4) {
    //     elapsed += gf_1d_gpu(&e, GAUSSIAN_3x3_load_balance12_local);
    // }

    // for (cv::Mat& e : output5) {
    //     elapsed += gf_1d_gpu(&e, GAUSSIAN_3x3_load_balance16_local);
    // }



    // for (cv::Mat& e : output1) {
    //     elapsed += gf_1d_gpu(&e, GAUSSIAN_3x3_vectorized2_local);
    // }

    // for (cv::Mat& e : output2) {
    //     elapsed += gf_1d_gpu(&e, GAUSSIAN_3x3_vectorized4_local);
    // }

    // for (cv::Mat& e : output3) {
    //     elapsed += gf_1d_gpu(&e, GAUSSIAN_3x3_vectorized8_local);
    // }

    // for (cv::Mat& e : output4) {
    //     elapsed += gf_1d_gpu(&e, GAUSSIAN_3x3_vectorized12_local);
    // }

    // for (cv::Mat& e : output5) {
    //     elapsed += gf_1d_gpu(&e, GAUSSIAN_3x3_vectorized16_local);
    // }


    // for (cv::Mat& e : output1) {
    //     elapsed += gf_1d_gpu(&e, GAUSSIAN_3x3_constant);
    // }

    // for (cv::Mat& e : output1) {
    //     elapsed += gf_1d_gpu(&e, GAUSSIAN_3x3_load_balance2_constant);
    // }

    // for (cv::Mat& e : output2) {
    //     elapsed += gf_1d_gpu(&e, GAUSSIAN_3x3_load_balance4_constant);
    // }

    // for (cv::Mat& e : output3) {
    //     elapsed += gf_1d_gpu(&e, GAUSSIAN_3x3_load_balance8_constant);
    // }

    // for (cv::Mat& e : output4) {
    //     elapsed += gf_1d_gpu(&e, GAUSSIAN_3x3_load_balance12_constant);
    // }

    // for (cv::Mat& e : output5) {
    //     elapsed += gf_1d_gpu(&e, GAUSSIAN_3x3_load_balance16_constant);
    // }



    // for (cv::Mat& e : output1) {
    //     elapsed += gf_1d_gpu(&e, GAUSSIAN_3x3_vectorized2_constant);
    // }

    // for (cv::Mat& e : output2) {
    //     elapsed += gf_1d_gpu(&e, GAUSSIAN_3x3_vectorized4_constant);
    // }

    // for (cv::Mat& e : output3) {
    //     elapsed += gf_1d_gpu(&e, GAUSSIAN_3x3_vectorized8_constant);
    // }

    // for (cv::Mat& e : output4) {
    //     elapsed += gf_1d_gpu(&e, GAUSSIAN_3x3_vectorized12_constant);
    // }

    // for (cv::Mat& e : output5) {
    //     elapsed += gf_1d_gpu(&e, GAUSSIAN_3x3_vectorized16_constant);
    // }



    // for (cv::Mat& e : output1) {
    //     elapsed += gf_1d_gpu(&e, GAUSSIAN_3x3_shared);
    // }

    // for (cv::Mat& e : output1) {
    //     elapsed += gf_1d_gpu(&e, GAUSSIAN_3x3_load_balance2_shared);
    // }

    // for (cv::Mat& e : output2) {
    //     elapsed += gf_1d_gpu(&e, GAUSSIAN_3x3_load_balance4_shared);
    // }

    // for (cv::Mat& e : output3) {
    //     elapsed += gf_1d_gpu(&e, GAUSSIAN_3x3_load_balance8_shared);
    // }

    // for (cv::Mat& e : output4) {
    //     elapsed += gf_1d_gpu(&e, GAUSSIAN_3x3_load_balance12_shared);
    // }

    // for (cv::Mat& e : output5) {
    //     elapsed += gf_1d_gpu(&e, GAUSSIAN_3x3_load_balance16_shared);
    // }



    // for (cv::Mat& e : output1) {
    //     elapsed += gf_1d_gpu(&e, GAUSSIAN_3x3_vectorized2_shared);
    // }

    // for (cv::Mat& e : output2) {
    //     elapsed += gf_1d_gpu(&e, GAUSSIAN_3x3_vectorized4_shared);
    // }

    // for (cv::Mat& e : output3) {
    //     elapsed += gf_1d_gpu(&e, GAUSSIAN_3x3_vectorized8_shared);
    // }

    // for (cv::Mat& e : output4) {
    //     elapsed += gf_1d_gpu(&e, GAUSSIAN_3x3_vectorized12_shared);
    // }

    // for (cv::Mat& e : output5) {
    //     elapsed += gf_1d_gpu(&e, GAUSSIAN_3x3_vectorized16_shared);
    // }

    //cv::imshow("Output Image1 ", output1.at(0));
    //cv::imshow("Output Image2 ", output2.at(1));
    //cv::imshow("Output Image3 ", output3.at(1));
    //cv::imshow("Output Image4 ", output4.at(1));
    //cv::imshow("Output Image5 ", output5.at(1));
    //cv::imshow("Output Image6 ", output6.at(1));


    //cv::imwrite("test.jpg", output1.at(0));

    cv::waitKey(0);
    return 0;
}
