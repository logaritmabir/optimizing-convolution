#include <gtest/gtest.h>
#include "kernels.cuh"

class KernelTest : public ::testing::Test {
protected:
    void SetUp() override {
        rows = 32;
        cols = 32;
        inputImg = cv::Mat(rows, cols, CV_8UC1, cv::Scalar(1));
        outputImg = cv::Mat(rows, cols, CV_8UC1, cv::Scalar(0));
    }

    int rows;
    int cols;
    cv::Mat inputImg;
    cv::Mat outputImg;
};

TEST_F(KernelTest, GM_3x3) {
    callKernel(&inputImg, &outputImg, KernelType::GM_3x3);
    ASSERT_NEAR(outputImg.at<unsigned char>(16, 16), 1, 1);
}

TEST_F(KernelTest, CM_3x3) {
    callKernel(&inputImg, &outputImg, KernelType::CM_3x3);
    ASSERT_NEAR(outputImg.at<unsigned char>(16, 16), 1, 1);
}

TEST_F(KernelTest, SM_3x3) {
    callKernel(&inputImg, &outputImg, KernelType::SM_3x3);
    ASSERT_NEAR(outputImg.at<unsigned char>(16, 16), 1, 1);
}

TEST_F(KernelTest, GM_3x3_CF2) {
    callKernel(&inputImg, &outputImg, KernelType::GM_3x3_CF2);
    ASSERT_NEAR(outputImg.at<unsigned char>(16, 16), 1, 1);
}

TEST_F(KernelTest, GM_3x3_CF4) {
    callKernel(&inputImg, &outputImg, KernelType::GM_3x3_CF4);
    ASSERT_NEAR(outputImg.at<unsigned char>(16, 16), 1, 1);
}

TEST_F(KernelTest, GM_3x3_CF8) {
    callKernel(&inputImg, &outputImg, KernelType::GM_3x3_CF8);
    ASSERT_NEAR(outputImg.at<unsigned char>(16, 16), 1, 1);
}

TEST_F(KernelTest, GM_3x3_CF12) {
    callKernel(&inputImg, &outputImg, KernelType::GM_3x3_CF12);
    ASSERT_NEAR(outputImg.at<unsigned char>(16, 16), 1, 1);
}

TEST_F(KernelTest, GM_3x3_CF16) {
    callKernel(&inputImg, &outputImg, KernelType::GM_3x3_CF16);
    ASSERT_NEAR(outputImg.at<unsigned char>(16, 16), 1, 1);
}

TEST_F(KernelTest, CM_3x3_CF2) {
    callKernel(&inputImg, &outputImg, KernelType::CM_3x3_CF2);
    ASSERT_NEAR(outputImg.at<unsigned char>(16, 16), 1, 1);
}

TEST_F(KernelTest, CM_3x3_CF4) {
    callKernel(&inputImg, &outputImg, KernelType::CM_3x3_CF4);
    ASSERT_NEAR(outputImg.at<unsigned char>(16, 16), 1, 1);
}

TEST_F(KernelTest, CM_3x3_CF8) {
    callKernel(&inputImg, &outputImg, KernelType::CM_3x3_CF8);
    ASSERT_NEAR(outputImg.at<unsigned char>(16, 16), 1, 1);
}

TEST_F(KernelTest, CM_3x3_CF12) {
    callKernel(&inputImg, &outputImg, KernelType::CM_3x3_CF12);
    ASSERT_NEAR(outputImg.at<unsigned char>(16, 16), 1, 1);
}

TEST_F(KernelTest, CM_3x3_CF16) {
    callKernel(&inputImg, &outputImg, KernelType::CM_3x3_CF16);
    ASSERT_NEAR(outputImg.at<unsigned char>(16, 16), 1, 1);
}

TEST_F(KernelTest, GM_3x3_CF2_Vec) {
    callKernel(&inputImg, &outputImg, KernelType::GM_3x3_CF2_Vec);
    ASSERT_NEAR(outputImg.at<unsigned char>(16, 16), 1, 1);
}

TEST_F(KernelTest, GM_3x3_CF4_Vec) {
    callKernel(&inputImg, &outputImg, KernelType::GM_3x3_CF4_Vec);
    ASSERT_NEAR(outputImg.at<unsigned char>(16, 16), 1, 1);
}

TEST_F(KernelTest, GM_3x3_CF8_Vec) {
    callKernel(&inputImg, &outputImg, KernelType::GM_3x3_CF8_Vec);
    ASSERT_NEAR(outputImg.at<unsigned char>(16, 16), 1, 1);
}

TEST_F(KernelTest, GM_3x3_CF12_Vec) {
    callKernel(&inputImg, &outputImg, KernelType::GM_3x3_CF12_Vec);
    ASSERT_NEAR(outputImg.at<unsigned char>(16, 16), 1, 1);
}

TEST_F(KernelTest, GM_3x3_CF16_Vec) {
    callKernel(&inputImg, &outputImg, KernelType::GM_3x3_CF16_Vec);
    ASSERT_NEAR(outputImg.at<unsigned char>(16, 16), 1, 1);
}

TEST_F(KernelTest, CM_3x3_CF2_Vec) {
    callKernel(&inputImg, &outputImg, KernelType::CM_3x3_CF2_Vec);
    ASSERT_NEAR(outputImg.at<unsigned char>(16, 16), 1, 1);
}

TEST_F(KernelTest, CM_3x3_CF4_Vec) {
    callKernel(&inputImg, &outputImg, KernelType::CM_3x3_CF4_Vec);
    ASSERT_NEAR(outputImg.at<unsigned char>(16, 16), 1, 1);
}

TEST_F(KernelTest, CM_3x3_CF8_Vec) {
    callKernel(&inputImg, &outputImg, KernelType::CM_3x3_CF8_Vec);
    ASSERT_NEAR(outputImg.at<unsigned char>(16, 16), 1, 1);
}

TEST_F(KernelTest, CM_3x3_CF12_Vec) {
    callKernel(&inputImg, &outputImg, KernelType::CM_3x3_CF12_Vec);
    ASSERT_NEAR(outputImg.at<unsigned char>(16, 16), 1, 1);
}

TEST_F(KernelTest, CM_3x3_CF16_Vec) {
    callKernel(&inputImg, &outputImg, KernelType::CM_3x3_CF16_Vec);
    ASSERT_NEAR(outputImg.at<unsigned char>(16, 16), 1, 1);
}

TEST_F(KernelTest, SM_3x3_CF2) {
    callKernel(&inputImg, &outputImg, KernelType::SM_3x3_CF2);
    ASSERT_NEAR(outputImg.at<unsigned char>(16, 16), 1, 1);
}

TEST_F(KernelTest, SM_3x3_CF4) {
    callKernel(&inputImg, &outputImg, KernelType::SM_3x3_CF4);
    ASSERT_NEAR(outputImg.at<unsigned char>(16, 16), 1, 1);
}

TEST_F(KernelTest, SM_3x3_CF8) {
    callKernel(&inputImg, &outputImg, KernelType::SM_3x3_CF8);
    ASSERT_NEAR(outputImg.at<unsigned char>(16, 16), 1, 1);
}

TEST_F(KernelTest, SM_3x3_CF12) {
    callKernel(&inputImg, &outputImg, KernelType::SM_3x3_CF12);
    ASSERT_NEAR(outputImg.at<unsigned char>(16, 16), 1, 1);
}

TEST_F(KernelTest, SM_3x3_CF16) {
    callKernel(&inputImg, &outputImg, KernelType::SM_3x3_CF16);
    ASSERT_NEAR(outputImg.at<unsigned char>(16, 16), 1, 1);
}

TEST_F(KernelTest, SM_3x3_CF2_Vec) {
    callKernel(&inputImg, &outputImg, KernelType::SM_3x3_CF2_Vec);
    ASSERT_NEAR(outputImg.at<unsigned char>(16, 16), 1, 1);
}

TEST_F(KernelTest, SM_3x3_CF4_Vec) {
    callKernel(&inputImg, &outputImg, KernelType::SM_3x3_CF4_Vec);
    ASSERT_NEAR(outputImg.at<unsigned char>(16, 16), 1, 1);
}

TEST_F(KernelTest, SM_3x3_CF8_Vec) {
    callKernel(&inputImg, &outputImg, KernelType::SM_3x3_CF8_Vec);
    ASSERT_NEAR(outputImg.at<unsigned char>(16, 16), 1, 1);
}

TEST_F(KernelTest, SM_3x3_CF12_Vec) {
    callKernel(&inputImg, &outputImg, KernelType::SM_3x3_CF12_Vec);
    ASSERT_NEAR(outputImg.at<unsigned char>(16, 16), 1, 1);
}

TEST_F(KernelTest, SM_3x3_CF16_Vec) {
    callKernel(&inputImg, &outputImg, KernelType::SM_3x3_CF16_Vec);
    ASSERT_NEAR(outputImg.at<unsigned char>(16, 16), 1, 1);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
