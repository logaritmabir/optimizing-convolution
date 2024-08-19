#include "histogram.cuh"
#include "gaussian.cuh"
#include "gamma.cuh"

float gc_3d_cppthreads(cv::Mat inputImg, cv::Mat* outputImg, float gamma) {
	unsigned char* input = inputImg.data;
	unsigned char* output = outputImg->data;

	unsigned int rows = inputImg.rows;
	unsigned int cols = inputImg.cols * 3;

	auto start = std::chrono::steady_clock::now();

	std::vector <std::thread> threads;
	const int MAX_THREAD_SUPPORT = std::thread::hardware_concurrency();

	int stride = rows / MAX_THREAD_SUPPORT;

	unsigned char LUT[256] = { 0 };

	for (int i = 0; i < 256; i++) {
		LUT[i] = static_cast<uchar>(pow(i / 255.0f, gamma) * 255);
	}

	for (int i = 0; i < MAX_THREAD_SUPPORT; i++) {
		threads.push_back(std::thread([&, i]() {
			int range_start = stride * i;
			int range_end = (i == MAX_THREAD_SUPPORT - 1) ? rows : stride * (i + 1);

			for (int x = range_start; x < range_end; x++) {
				for (int y = 0; y < cols; y++) {
					int index = x * cols + y;
					output[index] = LUT[input[index]];
				}
			}
			}));
	}
	for (std::thread& thread : threads) {
		thread.join();
	}

	auto end = std::chrono::steady_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start) / 1000.0f;
	return elapsed.count();
}

float gc_3d_openmp(cv::Mat inputImg, cv::Mat* outputImg, float gamma) {
	unsigned char* input = inputImg.data;
	unsigned char* output = outputImg->data;

	unsigned int rows = inputImg.rows;
	unsigned int cols = inputImg.cols * 3;

	auto start = std::chrono::steady_clock::now();

	unsigned char LUT[256] = { 0 };

#pragma omp parallel for
	for (int i = 0; i < 256; i++) {
		LUT[i] = static_cast<unsigned char>(pow(i / 255.0f, gamma) * 255);
	}

#pragma omp parallel for
	for (int x = 0; x < rows; x++) {
		for (int y = 0; y < cols; y++) {
			int index = x * cols + y;
			output[index] = LUT[input[index]];
		}
	}

	auto end = std::chrono::steady_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start) / 1000.0f;
	return elapsed.count();
}

float he_1d_cppthreads(cv::Mat inputImg, cv::Mat* outputImg) {
	const unsigned char* input = inputImg.data;
	unsigned char* output = outputImg->data;

	const unsigned int rows = inputImg.rows;
	const unsigned int cols = inputImg.cols;

	int histogram[256] = { 0 };
	float normalizedHistogram[256] = { 0 };
	float cdf[256] = { 0 };
	int equalization[256] = { 0 };
	int pixels = cols * rows;

	std::vector <std::thread> threads;
	std::mutex mtx;
	std::condition_variable cv;

	const int MAX_THREAD_SUPPORT = 12;
	const int stride = rows / MAX_THREAD_SUPPORT;
	const int stride_for_256 = 256 / MAX_THREAD_SUPPORT;

	int step1_count = 0;
	int step2_count = 0;
	int step3_count = 0;
	int step4_count = 0;
	auto start = std::chrono::steady_clock::now();

	for (int id = 0; id < MAX_THREAD_SUPPORT; id++) {
		threads.push_back(std::thread([&, id]() {
			int range_start = stride * id;
			int range_end = (id == MAX_THREAD_SUPPORT - 1) ? rows : stride * (id + 1);

			int t_histogram[256] = { 0 };

			for (int r = range_start; r < range_end; r++) {
				for (int c = 0; c < cols; c++) {
					{
						t_histogram[input[r * cols + c]]++;
					}
				}
			}

			{
				std::unique_lock<std::mutex> lck(mtx);
				for (int i = 0; i < 256; i++) {
					histogram[i] += t_histogram[i];
				}
			}

			{
				std::unique_lock<std::mutex> lck(mtx);
				if (++step1_count == MAX_THREAD_SUPPORT) {
					cv.notify_all();
				}
				else {
					cv.wait(lck);
				}
			}

			range_start = stride_for_256 * id;
			range_end = (id == MAX_THREAD_SUPPORT - 1) ? 256 : stride_for_256 * (id + 1);

			for (int i = range_start; i < range_end; i++) {
				normalizedHistogram[i] = histogram[i] / (float)pixels;
			}
			{
				std::unique_lock<std::mutex> lck(mtx);
				if (++step2_count == MAX_THREAD_SUPPORT) {
					cv.notify_all();
				}
				else {
					cv.wait(lck);
				}
			}
			cdf[0] = normalizedHistogram[0];

			for (int i = range_start; i < range_end; i++) {
				if (i == 0)
					continue;
				float sum = 0.0f;
				for (int j = 0; j <= i; j++) {
					sum += normalizedHistogram[j];
				}
				cdf[i] = sum;
			}
			{
				std::unique_lock<std::mutex> lck(mtx);
				if (++step3_count == MAX_THREAD_SUPPORT) {
					cv.notify_all();
				}
				else {
					cv.wait(lck);
				}
			}

			for (int i = range_start; i < range_end; i++) {
				equalization[i] = int((cdf[i] * 255.0f) + 0.5f);
			}

			{
				std::unique_lock<std::mutex> lck(mtx);
				if (++step4_count == MAX_THREAD_SUPPORT) {
					cv.notify_all();
				}
				else {
					cv.wait(lck);
				}
			}

			range_start = stride * id;
			range_end = (id == MAX_THREAD_SUPPORT - 1) ? rows : stride * (id + 1);
			for (int r = range_start; r < range_end; r++) {
				for (int c = 0; c < cols; c++) {
					int index = r * cols + c;
					output[index] = equalization[input[index]];
				}
			}
			}));
	}
	for (std::thread& thread : threads) {
		thread.join();
	}
	auto end = std::chrono::steady_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start) / 1000.0f;
	return elapsed.count();
}

float he_3d_cppthreads(cv::Mat inputImg, cv::Mat* outputImg) {
	const unsigned char* input = inputImg.data;
	unsigned char* output = outputImg->data;

	const unsigned int rows = inputImg.rows;
	const unsigned int cols = inputImg.cols;

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
	int pixels = cols * rows;

	std::vector <std::thread> threads;
	std::mutex mtx;
	std::condition_variable cv;

	const int MAX_THREAD_SUPPORT = 12;
	const int stride = rows / MAX_THREAD_SUPPORT;
	const int stride_for_256 = 256 / MAX_THREAD_SUPPORT;

	int step1_count = 0;
	int step2_count = 0;
	int step3_count = 0;
	int step4_count = 0;
	auto start = std::chrono::steady_clock::now();

	for (int id = 0; id < MAX_THREAD_SUPPORT; id++) {
		threads.push_back(std::thread([&, id]() {
			int range_start = stride * id;
			int range_end = (id == MAX_THREAD_SUPPORT - 1) ? rows : stride * (id + 1);

			int local_histogram_red[256] = { 0 };
			int local_histogram_green[256] = { 0 };
			int local_histogram_blue[256] = { 0 };

			for (int r = range_start; r < range_end; r++) {
				for (int c = 0; c < cols; c++) {
					{
						int index = (r * cols + c) * 3;
						local_histogram_red[input[index]]++;
						local_histogram_green[input[index + 1]]++;
						local_histogram_blue[input[index + 2]]++;
					}
				}
			}
			{
				std::unique_lock<std::mutex> lck(mtx);
				for (int i = 0; i < 256; i++) {
					histogram_red[i] += local_histogram_red[i];
					histogram_green[i] += local_histogram_green[i];
					histogram_blue[i] += local_histogram_blue[i];
				}
			}

			{
				std::unique_lock<std::mutex> lck(mtx);
				if (++step1_count == MAX_THREAD_SUPPORT) {
					cv.notify_all();
				}
				else {
					cv.wait(lck);
				}
			}

			range_start = stride_for_256 * id;
			range_end = (id == MAX_THREAD_SUPPORT - 1) ? 256 : stride_for_256 * (id + 1);

			for (int i = range_start; i < range_end; i++) {
				normalize_histogram_red[i] = histogram_red[i] / (float)pixels;
				normalize_histogram_green[i] = histogram_green[i] / (float)pixels;
				normalize_histogram_blue[i] = histogram_blue[i] / (float)pixels;
			}
			{
				std::unique_lock<std::mutex> lck(mtx);
				if (++step2_count == MAX_THREAD_SUPPORT) {
					cv.notify_all();
				}
				else {
					cv.wait(lck);
				}
			}
			cdf_red[0] = normalize_histogram_red[0];
			cdf_green[0] = normalize_histogram_green[0];
			cdf_blue[0] = normalize_histogram_blue[0];

			for (int i = range_start; i < range_end; i++) {
				float sum_red = 0;
				float sum_green = 0;
				float sum_blue = 0;
				for (int j = 0; j <= i; j++) {
					sum_red += normalize_histogram_red[j];
					sum_green += normalize_histogram_green[j];
					sum_blue += normalize_histogram_blue[j];
				}
				cdf_red[i] = sum_red;
				cdf_green[i] = sum_green;
				cdf_blue[i] = sum_blue;
			}
			{
				std::unique_lock<std::mutex> lck(mtx);
				if (++step3_count == MAX_THREAD_SUPPORT) {
					cv.notify_all();
				}
				else {
					cv.wait(lck);
				}
			}

			for (int i = range_start; i < range_end; i++) {
				equalization_red[i] = int((cdf_red[i] * 255.0f) + 0.5f);
				equalization_green[i] = int((cdf_green[i] * 255.0f) + 0.5f);
				equalization_blue[i] = int((cdf_blue[i] * 255.0f) + 0.5f);
			}

			{
				std::unique_lock<std::mutex> lck(mtx);
				if (++step4_count == MAX_THREAD_SUPPORT) {
					cv.notify_all();
				}
				else {
					cv.wait(lck);
				}
			}

			range_start = stride * id;
			range_end = (id == MAX_THREAD_SUPPORT - 1) ? rows : stride * (id + 1);
			for (int r = range_start; r < range_end; r++) {
				for (int c = 0; c < cols; c++) {
					int index = (r * cols + c) * 3;
					output[index] = equalization_red[input[index]];
					output[index + 1] = equalization_green[input[index + 1]];
					output[index + 2] = equalization_blue[input[index + 2]];
				}
			}
			}));
	}
	for (std::thread& thread : threads) {
		thread.join();
	}
	auto end = std::chrono::steady_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start) / 1000.0f;
	return elapsed.count();
}

float he_1d_openmp(cv::Mat inputImg, cv::Mat* outputImg) {
	unsigned char* input = inputImg.data;
	unsigned char* output = outputImg->data;

	int histogram[256] = { 0 };
	float cdf[256] = { 0 };
	float normalizedHistogram[256] = { 0 };
	int equalization[256] = { 0 };

	int pixels = inputImg.cols * inputImg.rows;

	auto start = std::chrono::steady_clock::now();

	omp_set_num_threads(12);
#pragma omp parallel
	{
		int l_histogram[256] = { 0 };

#pragma omp for
		for (int i = 0; i < pixels; i++) {
			l_histogram[input[i]]++;
		}


#pragma omp critical
		{
			for (int i = 0; i < 256; i++) {
				histogram[i] += l_histogram[i];
			}
		}
	}

#pragma omp parallel for
	for (int i = 0; i < 256; i++) {
		normalizedHistogram[i] = (histogram[i] / (float)pixels);
	}

	cdf[0] = normalizedHistogram[0];

	for (int i = 1; i < 256; i++) {
		cdf[i] = cdf[i - 1] + normalizedHistogram[i];
	}

#pragma omp parallel for 
	for (int i = 0; i < 256; i++) {
		equalization[i] = int((cdf[i] * 255.0f) + 0.5f);
	}

#pragma omp parallel for
	for (int i = 0; i < pixels; i++) {
		output[i] = equalization[input[i]];
	}

	auto end = std::chrono::steady_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start) / 1000.0f;
	return elapsed.count();
}

float he_3d_openmp(cv::Mat input_img, cv::Mat* output_img) {
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
#pragma omp parallel
	{
		int l_histogram_red[256] = { 0 };
		int l_histogram_green[256] = { 0 };
		int l_histogram_blue[256] = { 0 };
#pragma omp for
		for (int i = 0; i < data_size; i++) {
			int index = i * 3;
			l_histogram_red[input[index]]++;
			l_histogram_green[input[index + 1]]++;
			l_histogram_blue[input[index + 2]]++;
		}
#pragma omp critical
		{
			for (int i = 0; i < 256; i++) {
				histogram_red[i] += l_histogram_red[i];
				histogram_green[i] += l_histogram_green[i];
				histogram_blue[i] += l_histogram_blue[i];
			}
		}
	}

#pragma omp parallel for
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

#pragma omp parallel for
	for (int i = 0; i < 256; i++) {
		equalization_red[i] = int((cdf_red[i] * 255.0f) + 0.5f);
		equalization_green[i] = int((cdf_green[i] * 255.0f) + 0.5f);
		equalization_blue[i] = int((cdf_blue[i] * 255.0f) + 0.5f);
	}

#pragma omp parallel for
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


float gf_1d_cppthreads(cv::Mat input_img, cv::Mat* output_img)
{
	unsigned char* input = input_img.data;
	unsigned char* output = output_img->data;
	int cols = input_img.cols;
	int rows = input_img.rows;
	const unsigned short mask_dim = 3;
	unsigned char kernel[mask_dim][mask_dim] = { {1, 2, 1}, {2, 4, 2}, {1, 2, 1} };

	std::vector<std::thread> threads;
	const int MAX_THREAD_SUPPORT = std::thread::hardware_concurrency();

	int stride = rows / MAX_THREAD_SUPPORT;

	auto start = std::chrono::steady_clock::now();
	for (int i = 0; i < MAX_THREAD_SUPPORT; i++)
	{
		threads.push_back(std::thread([&, i]() {
			int range_start = stride * i;
			int range_end = (i == MAX_THREAD_SUPPORT - 1) ? cols : stride * (i + 1);

			for (int r = range_start; r < range_end; r++) { /*row loop*/
				for (int c = 0; c < cols; c++) { /*col loop*/
					if (r > 0 && r < rows - 1 && c > 0 && c < cols - 1) {
						int new_pixel_value = 0;
						for (int mr = 0; mr < mask_dim; mr++) { /*matrix row*/
							for (int mc = 0; mc < mask_dim; mc++) { /*matrix col*/
								int r_index = r + mr - 1;
								int c_index = c + mc - 1;
								new_pixel_value += input[r_index * cols + c_index] * kernel[mr][mc];
							}
						}
						output[r * cols + c] = static_cast<unsigned char>(new_pixel_value / 16);
					}
				}
			} }));
	}
	for (std::thread& th : threads)
	{
		th.join();
	}
	auto end = std::chrono::steady_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start) / 1000.0f;
	return elapsed.count();
}

float gf_1d_openmp(cv::Mat input_img, cv::Mat* output_img)
{
	int cols = input_img.cols;
	int rows = input_img.rows;

	unsigned char* input = input_img.data;
	unsigned char* output = output_img->data;
	const unsigned short mask_dim = 3;
	unsigned char kernel[mask_dim][mask_dim] = { {1, 2, 1}, {2, 4, 2}, {1, 2, 1} };
	auto start = std::chrono::steady_clock::now();

#pragma omp parallel for
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

float gf_3d_cppthreads(cv::Mat input_img, cv::Mat* output_img)
{
	unsigned char* input = input_img.data;
	unsigned char* output = output_img->data;
	int cols = input_img.cols;
	int rows = input_img.rows;
	const unsigned short mask_dim = 3;
	unsigned char kernel[mask_dim][mask_dim] = { {1, 2, 1}, {2, 4, 2}, {1, 2, 1} };

	std::vector<std::thread> threads;
	const int MAX_THREAD_SUPPORT = std::thread::hardware_concurrency();

	int stride = rows / MAX_THREAD_SUPPORT;

	auto start = std::chrono::steady_clock::now();
	for (int i = 0; i < MAX_THREAD_SUPPORT; i++)
	{
		threads.push_back(std::thread([&, i]()
			{
				int range_start = stride * i;
				int range_end = (i == MAX_THREAD_SUPPORT - 1) ? cols : stride * (i + 1);

				for (int r = range_start; r < range_end; r++) { /*row loop*/
					for (int c = 0; c < cols; c++) { /*col loop*/
						if (r > 0 && r < rows - 1 && c > 0 && c < cols - 1) {
							int new_pixel_value_red = 0;
							int new_pixel_value_green = 0;
							int new_pixel_value_blue = 0;
							for (int mr = 0; mr < mask_dim; mr++) { /*matrix row*/
								for (int mc = 0; mc < mask_dim; mc++) { /*matrix col*/
									int r_index = r + mr - 1;
									int c_index = c + mc - 1;
									new_pixel_value_red += input[(r_index * cols + c_index) * 3] * kernel[mr][mc];
									new_pixel_value_green += input[(r_index * cols + c_index) * 3 + 1] * kernel[mr][mc];
									new_pixel_value_blue += input[(r_index * cols + c_index) * 3 + 2] * kernel[mr][mc];

								}
							}
							output[(r * cols + c) * 3] = static_cast<unsigned char>(new_pixel_value_red / 16);
							output[(r * cols + c) * 3 + 1] = static_cast<unsigned char>(new_pixel_value_green / 16);
							output[(r * cols + c) * 3 + 2] = static_cast<unsigned char>(new_pixel_value_blue / 16);
						}
					}
				} }));
	}
	for (std::thread& th : threads)
	{
		th.join();
	}
	auto end = std::chrono::steady_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start) / 1000.0f;
	return elapsed.count();
}

float gf_3d_openmp(cv::Mat input_img, cv::Mat* output_img)
{
	int cols = input_img.cols;
	int rows = input_img.rows;

	unsigned char* input = input_img.data;
	unsigned char* output = output_img->data;
	const unsigned short mask_dim = 3;

	unsigned char kernel[mask_dim][mask_dim] = { {1, 2, 1}, {2, 4, 2}, {1, 2, 1} };
	auto start = std::chrono::steady_clock::now();

#pragma omp parallel for
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
