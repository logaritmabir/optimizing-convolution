#include "kernels.cuh"

int main(int argc, char** argv)
{
    cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT); /*suppressing annoying OpenCV infos in cmdline*/
    std::string img_path512 = "../images/inputs/512.png";
    std::string img_path2048 = "../images/inputs/2048.png";
    std::string img_path4096 = "../images/inputs/4096.png";
    std::string img_path8192 = "../images/inputs/8192.png";

    std::vector<cv::Mat> input_8UC1;
    std::vector<cv::Mat> input_32FC1;
    input_8UC1.push_back(cv::imread(img_path4096, cv::IMREAD_GRAYSCALE));

    for (cv::Mat& image : input_8UC1) {
        cv::Mat buffer;
        image.convertTo(buffer, CV_32FC1, 1.0 / 255.0);
        input_32FC1.push_back(buffer);
    }

    std::vector <cv::Mat> output_8UC1;
    std::vector <cv::Mat> output_32FC1;

    for(int i = 0; i < input_8UC1.size(); i++){
        output_8UC1.push_back(input_8UC1.at(i).clone());
        output_32FC1.push_back(input_32FC1.at(i).clone());
    }

    #ifdef IMTYPE_FLOAT
        for(int i = 0; i < input_32FC1.size(); i++){
            launchKernels(&input_32FC1.at(i), &output_32FC1.at(i));
        }
    #elif defined(IMTYPE_UCHAR)
        for(int i = 0; i < input_8UC1.size(); i++){
            launchKernels(&input_8UC1.at(i), &output_8UC1.at(i));
        }
    #endif
    return 0;
}