#include <string>
#include <cmath>
#include <chrono>
#include <math.h>

#include "kernels.cuh"

int main(int argc, char** argv)
{
    cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT); /*suppressing annoying OpenCV infos in cmdline*/
    std::string img_path4096 = "../images/4096.png";

    std::vector<cv::Mat> gray_input_images; /*input images in black and white*/
    
    gray_input_images.push_back(cv::imread(img_path4096, cv::IMREAD_GRAYSCALE));

    std::vector <cv::Mat> output1; /*image buffers that stores copy of gray images*/

    for (cv::Mat& e : gray_input_images) {
        output1.push_back(e.clone());
    }

    for(int i = 0; i < gray_input_images.size(); i++){
        launch_base_kernels(&gray_input_images.at(i),&output1.at(i));
    }

    return 0;
}