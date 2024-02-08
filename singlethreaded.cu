#include "histogram.cuh"
#include "gaussian.cuh"
#include "gamma.cuh"

float gc_3d_opencv(cv::Mat input_img, cv::Mat* output_img, float gamma) {
	auto start = std::chrono::steady_clock::now();

	cv::Mat lookUpTable(1, 256, CV_8U);
	uchar* p = lookUpTable.ptr();
	for (int i = 0; i < 256; ++i)
		p[i] = cv::saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);
	LUT(input_img, lookUpTable, *output_img);

	auto end = std::chrono::steady_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start) / 1000.0f;
	return elapsed.count();
}

float gc_3d_single_thread(cv::Mat inputImg, cv::Mat* outputImg, float gamma) {
	unsigned char* input = inputImg.data;
	unsigned char* output = outputImg->data;

	unsigned int rows = inputImg.rows;
	unsigned int cols = inputImg.cols;
	unsigned int pixels = rows * cols * 3;

	auto start = std::chrono::steady_clock::now();
	unsigned char LUT[256] = { 0 };
	for (int i = 0; i < 256; i++) {
		LUT[i] = pow(i / 255.0f, gamma) * 255;
	}
	for (int i = 0; i < pixels; i++) {
		output[i] = LUT[input[i]];
	}
	auto end = std::chrono::steady_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start) / 1000.0f;
	return elapsed.count();
}

float he_1d_single_thread(cv::Mat inputImg, cv::Mat* outputImg) {
	unsigned char* input = inputImg.data;
	unsigned char* output = outputImg->data;

	int histogram[256] = { 0 };
	float cdf[256] = { 0 };
	float normalizedHistogram[256] = { 0 };
	int equalization[256] = { 0 };

	int pixels = inputImg.cols * inputImg.rows;

	auto start = std::chrono::steady_clock::now();

	for (int i = 0; i < pixels; i++) {
		histogram[input[i]]++;
	}

	for (int i = 0; i < 256; i++) {
		normalizedHistogram[i] = (histogram[i] / (float)pixels);
	}

	cdf[0] = normalizedHistogram[0];
	for (int i = 1; i < 256; i++) {
		cdf[i] = cdf[i - 1] + normalizedHistogram[i];
	}

	for (int i = 0; i < 256; i++) {
		equalization[i] = int((cdf[i] * 255.0f) + 0.5f);
	}

	for (int i = 0; i < pixels; i++) {
		output[i] = equalization[input[i]];
	}

	auto end = std::chrono::steady_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start) / 1000.0f;
	return elapsed.count();
}

float he_3d_single_thread(cv::Mat input_img, cv::Mat* output_img) {
	unsigned char* input = input_img.data;
	unsigned char* output = output_img->data;

	int histogram_red[256] = { 0 };
	int histogram_green[256] = { 0 };
	int histogram_blue[256] = { 0 };

	float normalize_histogram_red[256] = { 0 };
	float normalize_histogram_green[256] = { 0 };
	float normalize_histogram_blue[256] = { 0 };

	float cdf_red[256] = { 0 };
	float cdf_green[256] = { 0 };
	float cdf_blue[256] = { 0 };

	int equalization_red[256] = { 0 };
	int equalization_green[256] = { 0 };
	int equalization_blue[256] = { 0 };

	int pixels = input_img.cols * input_img.rows;
	int data_size = pixels;

	auto start = std::chrono::steady_clock::now();

	for (int i = 0; i < data_size; i++) {
		int index = i * 3;

		histogram_red[input[index]]++;
		histogram_green[input[index + 1]]++;
		histogram_blue[input[index + 2]]++;
	}
	for (int i = 0; i < 256; i++) {
		normalize_histogram_red[i] = (histogram_red[i] / (float)pixels);
		normalize_histogram_green[i] = (histogram_green[i] / (float)pixels);
		normalize_histogram_blue[i] = (histogram_blue[i] / (float)pixels);
	}

	cdf_red[0] = normalize_histogram_red[0];
	cdf_green[0] = normalize_histogram_green[0];
	cdf_blue[0] = normalize_histogram_blue[0];

	for (int i = 1; i < 256; i++) {
		cdf_red[i] = cdf_red[i - 1] + normalize_histogram_red[i];
		cdf_green[i] = cdf_green[i - 1] + normalize_histogram_green[i];
		cdf_blue[i] = cdf_blue[i - 1] + normalize_histogram_blue[i];
	}

	for (int i = 0; i < 256; i++) {
		equalization_red[i] = int((cdf_red[i] * 255.0f) + 0.5f);
		equalization_green[i] = int((cdf_green[i] * 255.0f) + 0.5f);
		equalization_blue[i] = int((cdf_blue[i] * 255.0f) + 0.5f);
	}
	for (int i = 0; i < data_size; i++) {
		int index = i * 3;

		output[index] = equalization_red[input[index]];
		output[index + 1] = equalization_green[input[index + 1]];
		output[index + 2] = equalization_blue[input[index + 2]];
	}

	auto end = std::chrono::steady_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start) / 1000.0f;
	return elapsed.count();
}

float gf_1d_single_thread(cv::Mat input_img, cv::Mat* output_img)
{
	int cols = input_img.cols;
	int rows = input_img.rows;

	unsigned char* input = input_img.data;
	unsigned char* output = output_img->data;
	const unsigned short mask_dim = 3;
	unsigned char kernel[mask_dim][mask_dim] = { {1, 2, 1}, {2, 4, 2}, {1, 2, 1} };
	auto start = std::chrono::steady_clock::now();

	for (int i = 1; i < rows - 1; i++)
	{
		for (int j = 1; j < cols - 1; j++)
		{
			int newPixelValue = 0;
			for (int m = 0; m < mask_dim; m++)
			{
				for (int n = 0; n < mask_dim; n++)
				{
					newPixelValue += input[(i + m - 1) * cols + (j + n - 1)] * kernel[m][n];
				}
			}
			output[i * cols + j] = newPixelValue / 16;
		}
	}
	auto end = std::chrono::steady_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start) / 1000.0f;
	return elapsed.count();
}
float gf_3d_single_thread(cv::Mat input_img, cv::Mat* output_img)
{
	int cols = input_img.cols;
	int rows = input_img.rows;

	unsigned char* input = input_img.data;
	unsigned char* output = output_img->data;
	const unsigned short mask_dim = 3;

	unsigned char kernel[mask_dim][mask_dim] = { {1, 2, 1}, {2, 4, 2}, {1, 2, 1} };
	auto start = std::chrono::steady_clock::now();

	for (int i = 1; i < rows - 1; i++)
	{
		for (int j = 1; j < cols - 1; j++)
		{
			int new_red_val = 0;
			int new_green_val = 0;
			int new_blue_val = 0;
			for (int m = 0; m < mask_dim; m++)
			{
				for (int n = 0; n < mask_dim; n++)
				{
					new_red_val += input[(((i + m - 1) * cols + (j + n - 1))) * 3] * kernel[m][n];
					new_green_val += input[((i + m - 1) * cols + (j + n - 1)) * 3 + 1] * kernel[m][n];
					new_blue_val += input[((i + m - 1) * cols + (j + n - 1)) * 3 + 2] * kernel[m][n];
				}
			}
			output[(i * cols + j) * 3] = new_red_val / 16;
			output[(i * cols + j) * 3 + 1] = new_green_val / 16;
			output[(i * cols + j) * 3 + 2] = new_blue_val / 16;
		}
	}
	auto end = std::chrono::steady_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start) / 1000.0f;
	return elapsed.count();
}