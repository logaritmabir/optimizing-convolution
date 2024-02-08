#include "histogram.cuh"

__device__ int dev_histogram[256] = { 0 };
__device__ float dev_normalized_histogram[256] = { 0 };
__device__ float dev_cdf[256] = { 0 };
__device__ int dev_equalization_values[256] = { 0 };

/*color gpu variables*/

__device__ int dev_histogram_red[256] = { 0 };
__device__ float dev_normalized_histogram_red[256] = { 0 };
__device__ float dev_cdf_red[256] = { 0 };
__device__ int dev_equalization_values_red[256] = { 0 };

__device__ int dev_histogram_green[256] = { 0 };
__device__ float dev_normalized_histogram_green[256] = { 0 };
__device__ float dev_cdf_green[256] = { 0 };
__device__ int dev_equalization_values_green[256] = { 0 };

__device__ int dev_histogram_blue[256] = { 0 };
__device__ float dev_normalized_histogram_blue[256] = { 0 };
__device__ float dev_cdf_blue[256] = { 0 };
__device__ int dev_equalization_values_blue[256] = { 0 };

/*reduce branch variables*/

__device__ int dev_histogram_rb[256 * 3] = { 0 };
__device__ float dev_normalized_histogram_rb[256 * 3] = { 0 };
__device__ float dev_cdf_rb[256 * 3] = { 0 };
__device__ int dev_equalization_values_rb[256 * 3] = { 0 };

__global__ void k_1D_extract_histogram(unsigned char* input, int pixels) {
	int threadId = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (threadId >= pixels) {
		return;
	}

	atomicAdd(&dev_histogram[input[threadId]], 1);
}

__global__ void k_1D_normalize_cdf_equalization(int pixels) {
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	float sum = 0.0f;

	dev_normalized_histogram[threadId] = dev_histogram[threadId] / (float)(pixels);
	__syncthreads();

	for (int i = 0; i <= threadId; i++) {
		sum += dev_normalized_histogram[i];
	}
	dev_cdf[threadId] = sum;
	dev_equalization_values[threadId] = int((dev_cdf[threadId] * 255.0f) + 0.5f);
}

__global__ void k_1D_equalize(unsigned char* input, int pixels) {
	int threadId = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (threadId >= pixels) {
		return;
	}
	input[threadId] = static_cast<uchar>(dev_equalization_values[input[threadId]]);
}

__global__ void k_1D_extract_histogram_shared(unsigned char* input, int pixels) {
	__shared__ unsigned int cache[256];

	int thread_id_in_block = threadIdx.x;
	int thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (thread_id >= pixels) {
		return;
	}

	if (thread_id_in_block < 256) {
		cache[thread_id_in_block] = 0;
	}
	__syncthreads();

	atomicAdd(&cache[(input[thread_id])], 1);
	__syncthreads();

	if (thread_id_in_block < 256) {
		atomicAdd(&dev_histogram[thread_id_in_block], cache[thread_id_in_block]);
	}
}

__global__ void k_1D_normalize_cdf_equalization_shared(int pixels) {
	__shared__ float cache_normalized_histogram[256];
	__shared__ float cache_cdf[256];

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	cache_normalized_histogram[tid] = dev_histogram[tid] / (float)(pixels);
	__syncthreads();
	float sum = 0.0f;
	for (int i = 0; i <= tid; i++) {
		sum += cache_normalized_histogram[i];
	}
	cache_cdf[tid] = sum;
	dev_equalization_values[tid] = int((cache_cdf[tid] * 255.0f) + 0.5f);
}

__global__ void k_1D_equalize_shared(unsigned char* input, int pixels) { /*load the cache before threadId control*/
	__shared__ int cache_equalization_values[256];

	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	int cid = threadIdx.x;

	if (cid < 256) {
		cache_equalization_values[cid] = dev_equalization_values[cid];
	}
	__syncthreads();

	if (tid >= pixels) {
		return;
	}
	input[tid] = static_cast<uchar>(cache_equalization_values[input[tid]]);
}

__global__ void k_1D_equalize_shared_recompute(unsigned char* input, int pixels) { /*load the cache before threadId control*/
	__shared__ float s_normalized_histogram[256];
	__shared__ float s_cdf[256];
	__shared__ int s_equalization_values[256];

	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	int sid = threadIdx.x;

	if (sid < 256) {
		s_normalized_histogram[sid] = dev_histogram[sid] / (float)pixels;
		__syncthreads();
		float sum = 0.0f;
		for (int i = 0; i <= sid; i++) {
			sum += s_normalized_histogram[i];
		}
		s_cdf[sid] = sum;
		s_equalization_values[sid] = int((s_cdf[sid] * 255.0f) + 0.5f);
	}
	__syncthreads();

	if (tid >= pixels) {
		return;
	}

	input[tid] = static_cast<uchar>(s_equalization_values[input[tid]]);
}

float he_1d_gpu(cv::Mat input_img, cv::Mat* output_img, HISTOGRAM ver) {
	unsigned char* gpu_input = nullptr;

	unsigned char* input = input_img.data;
	unsigned char* output = output_img->data;

	unsigned int cols = input_img.cols;
	unsigned int rows = input_img.rows;

	unsigned int pixels = cols * rows;
	unsigned long int size = pixels * sizeof(unsigned char);

	CHECK_CUDA_ERROR(cudaHostRegister(output, size, 0));

	CHECK_CUDA_ERROR(cudaMalloc((unsigned char**)&gpu_input, size));
	CHECK_CUDA_ERROR(cudaMemcpy(gpu_input, output, size, cudaMemcpyHostToDevice));

	dim3 block(32 * 32);
	dim3 grid((pixels + block.x - 1) / block.x);

	switch (ver)
	{
	default:
		break;
	case HISTOGRAM_default: /*default*/
		k_1D_extract_histogram << <grid, block >> > (gpu_input, pixels);
		k_1D_normalize_cdf_equalization << <8, 32 >> > (pixels);
		k_1D_equalize << <grid, block >> > (gpu_input, pixels);
		break;
	case HISTOGRAM_shared: /*shared*/
		k_1D_extract_histogram_shared << <grid, block >> > (gpu_input, pixels);
		k_1D_normalize_cdf_equalization_shared << <1, 256 >> > (pixels);
		k_1D_equalize_shared << <grid, block >> > (gpu_input, pixels);
		break;
	case HISTOGRAM_recompute: /*shared recompute*/
		k_1D_extract_histogram_shared << <grid, block >> > (gpu_input, pixels);
		k_1D_equalize_shared_recompute << <grid, block >> > (gpu_input, pixels);
		break;
	}
	CHECK_CUDA_ERROR(cudaMemcpy(output, gpu_input, size, cudaMemcpyDeviceToHost));

	cudaFree(gpu_input);
	cudaFreeHost(output);
	cudaDeviceReset();
	return 0;
}

__global__ void k_3D_extract_histogram(unsigned char* input, int total_channel_size) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (tid >= total_channel_size) {
		return;
	}

	switch (tid % 3)
	{
	case 0:
		atomicAdd(&dev_histogram_red[input[tid]], 1);
		break;
	case 1:
		atomicAdd(&dev_histogram_green[input[tid]], 1);
		break;
	case 2:
		atomicAdd(&dev_histogram_blue[input[tid]], 1);
		break;
	default:
		break;
	}
}

__global__ void k_3D_extract_histogram_rb(unsigned char* input, int total_channel_size) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (tid < total_channel_size) {
		int pixel_value = input[tid];
		int color_index = tid % 3;
		int histogram_index = (color_index * 256) + pixel_value;
		atomicAdd(&dev_histogram_rb[histogram_index], 1);
	}
}

__global__ void k_3D_extract_histogram_rb_shared(unsigned char* input, int total_channel_size) {
	__shared__ unsigned int s_histogram[256 * 3];
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	int thread_id_in_block = threadIdx.x;

	if (tid < total_channel_size) {
		int pixel_value = input[tid];
		int color_index = tid % 3;
		int histogram_index = (color_index * 256) + pixel_value;
		atomicAdd(&s_histogram[histogram_index], 1);
		__syncthreads();
		if (thread_id_in_block < 256) {
			atomicAdd(&dev_histogram_rb[thread_id_in_block], s_histogram[thread_id_in_block]);
		}
	}
}

__global__ void k_3D_extract_histogram_shared(unsigned char* input, int total_pixel_size) {
	__shared__ unsigned int cache_histogram_red[256];
	__shared__ unsigned int cache_histogram_green[256];
	__shared__ unsigned int cache_histogram_blue[256];

	int thread_id_in_block = threadIdx.x;
	int thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (thread_id_in_block < 256) {
		cache_histogram_red[thread_id_in_block] = 0;
		cache_histogram_green[thread_id_in_block] = 0;
		cache_histogram_blue[thread_id_in_block] = 0;
	}
	__syncthreads();

	if (thread_id >= total_pixel_size) {
		return;
	}

	switch (thread_id % 3)
	{
	case 0:
		atomicAdd(&cache_histogram_red[(input[thread_id])], 1);
		break;
	case 1:
		atomicAdd(&cache_histogram_green[(input[thread_id])], 1);
		break;
	case 2:
		atomicAdd(&cache_histogram_blue[(input[thread_id])], 1);
	default:
		break;
	}

	__syncthreads();

	if (thread_id_in_block < 256) {
		atomicAdd(&dev_histogram_red[thread_id_in_block], cache_histogram_red[thread_id_in_block]);
		atomicAdd(&dev_histogram_green[thread_id_in_block], cache_histogram_green[thread_id_in_block]);
		atomicAdd(&dev_histogram_blue[thread_id_in_block], cache_histogram_blue[thread_id_in_block]);
	}
}

__global__ void k_3D_normalize_cdf_equalization(int pixels) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	dev_normalized_histogram_red[tid] = dev_histogram_red[tid] / (float)(pixels);
	dev_normalized_histogram_green[tid] = dev_histogram_green[tid] / (float)(pixels);
	dev_normalized_histogram_blue[tid] = dev_histogram_blue[tid] / (float)(pixels);
	__syncthreads();

	float sum_red = 0.0f, sum_green = 0.0f, sum_blue = 0.0f;
	for (int i = 0; i <= tid; i++) {
		sum_red += dev_normalized_histogram_red[i];
		sum_green += dev_normalized_histogram_green[i];
		sum_blue += dev_normalized_histogram_blue[i];
	}
	dev_cdf_red[tid] = sum_red;
	dev_cdf_green[tid] = sum_green;
	dev_cdf_blue[tid] = sum_blue;
	__syncthreads();

	dev_equalization_values_red[tid] = int((dev_cdf_red[tid] * 255.0f) + 0.5f);
	dev_equalization_values_green[tid] = int((dev_cdf_green[tid] * 255.0f) + 0.5f);
	dev_equalization_values_blue[tid] = int((dev_cdf_blue[tid] * 255.0f) + 0.5f);
}

__global__ void k_3D_normalize_cdf_equalization_rb(int pixels) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	dev_normalized_histogram_rb[tid] = dev_histogram_rb[tid] / (float)(pixels);

	__syncthreads();

	float sum = 0.0f;
	int color_index = tid / 256;
	int array_index_start = (color_index * 256);

	for (int i = array_index_start; i <= tid; i++) {
		sum += dev_normalized_histogram_rb[i];
	}
	dev_cdf_rb[tid] = sum;
	__syncthreads();

	dev_equalization_values_rb[tid] = (dev_cdf_rb[tid] * 255.0f) + 0.5f; /*int'e d�n��t�rmeyi sildim byurada*/
}


__global__ void k_3D_normalize_cdf_equalization_shared(int pixels) {
	__shared__ float cache_normalized_histogram_red[256];
	__shared__ float cache_normalized_histogram_green[256];
	__shared__ float cache_normalized_histogram_blue[256];

	__shared__ float cache_cdf_red[256];
	__shared__ float cache_cdf_green[256];
	__shared__ float cache_cdf_blue[256];

	int thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;

	cache_normalized_histogram_red[thread_id] = dev_histogram_red[thread_id] / (float)(pixels);
	cache_normalized_histogram_green[thread_id] = dev_histogram_green[thread_id] / (float)(pixels);
	cache_normalized_histogram_blue[thread_id] = dev_histogram_blue[thread_id] / (float)(pixels);
	__syncthreads();

	float sum_red = 0.0f, sum_green = 0.0f, sum_blue = 0.0f;
	for (int i = 0; i <= thread_id; i++) {
		sum_red += cache_normalized_histogram_red[i];
		sum_green += cache_normalized_histogram_green[i];
		sum_blue += cache_normalized_histogram_blue[i];
	}
	cache_cdf_red[thread_id] = sum_red;
	cache_cdf_green[thread_id] = sum_green;
	cache_cdf_blue[thread_id] = sum_blue;

	dev_equalization_values_red[thread_id] = int((cache_cdf_red[thread_id] * 255.0f) + 0.5f);
	dev_equalization_values_green[thread_id] = int((cache_cdf_green[thread_id] * 255.0f) + 0.5f);
	dev_equalization_values_blue[thread_id] = int((cache_cdf_blue[thread_id] * 255.0f) + 0.5f);
}

__global__ void k_3D_equalize(unsigned char* input, int total_channel_size) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (tid >= total_channel_size) {
		return;
	}

	switch (tid % 3)
	{
	case 0:
		input[tid] = static_cast<uchar>(dev_equalization_values_red[input[tid]]);
		break;
	case 1:
		input[tid] = static_cast<uchar>(dev_equalization_values_green[input[tid]]);
		break;
	case 2:
		input[tid] = static_cast<uchar>(dev_equalization_values_blue[input[tid]]);
		break;
	default:
		break;
	}
}

__global__ void k_3D_equalize_rb(unsigned char* input, int total_channel_size) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (tid >= total_channel_size) {
		return;
	}
	int color_index = tid % 3;
	input[tid] = static_cast<uchar>(dev_equalization_values_rb[color_index * 256 + input[tid]]);
}

__global__ void k_3D_equalize_shared(unsigned char* input, int total_pixel_size) {
	__shared__ int cache_equalization_values_red[256];
	__shared__ int cache_equalization_values_green[256];
	__shared__ int cache_equalization_values_blue[256];

	int thread_id_in_block = threadIdx.x;
	int thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (thread_id_in_block < 256) {
		cache_equalization_values_red[thread_id_in_block] = dev_equalization_values_red[thread_id_in_block];
		cache_equalization_values_green[thread_id_in_block] = dev_equalization_values_green[thread_id_in_block];
		cache_equalization_values_blue[thread_id_in_block] = dev_equalization_values_blue[thread_id_in_block];
	}

	if (thread_id >= total_pixel_size) {
		return;
	}
	__syncthreads();

	switch (thread_id % 3)
	{
	case 0:
		input[thread_id] = static_cast<uchar>(cache_equalization_values_red[input[thread_id]]);
		break;
	case 1:
		input[thread_id] = static_cast<uchar>(cache_equalization_values_green[input[thread_id]]);
		break;
	case 2:
		input[thread_id] = static_cast<uchar>(cache_equalization_values_blue[input[thread_id]]);
		break;

	default:
		break;
	}
}

float he_3d_gpu(cv::Mat input_img, cv::Mat* output_img, HISTOGRAM ver) {
	unsigned char* gpu_input = nullptr;

	unsigned char* input = input_img.data;
	unsigned char* output = output_img->data;

	unsigned int cols = input_img.cols;
	unsigned int rows = input_img.rows;

	unsigned int pixels = cols * rows;
	unsigned int total_channel_size = pixels * 3;
	unsigned long int size = cols * rows * sizeof(unsigned char) * 3;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	CHECK_CUDA_ERROR(cudaHostRegister(output, size, 0));

	CHECK_CUDA_ERROR(cudaMalloc((unsigned char**)&gpu_input, size));
	CHECK_CUDA_ERROR(cudaMemcpy(gpu_input, output, size, cudaMemcpyHostToDevice));

	dim3 block(32 * 32);
	dim3 grid((total_channel_size + block.x - 1) / block.x);

	switch (ver)
	{
	case HISTOGRAM_default:
		k_3D_extract_histogram << <grid, block >> > (gpu_input, total_channel_size);
		k_3D_normalize_cdf_equalization << <4, 64 >> > (pixels);
		k_3D_equalize << <grid, block >> > (gpu_input, total_channel_size);
		break;
	case HISTOGRAM_shared:
		k_3D_extract_histogram_shared << <grid, block >> > (gpu_input, total_channel_size);
		k_3D_normalize_cdf_equalization_shared << <1, 256 >> > (pixels);
		k_3D_equalize_shared << <grid, block >> > (gpu_input, total_channel_size);
		break;
	case HISTOGRAM_reduce_branches:
		k_3D_extract_histogram_rb << <grid, block >> > (gpu_input, total_channel_size);
		k_3D_normalize_cdf_equalization_rb << <12, 64 >> > (pixels);
		k_3D_equalize_rb << <grid, block >> > (gpu_input, total_channel_size);
		break;
	case HISTOGRAM_recompute:
		k_3D_extract_histogram << <grid, block >> > (gpu_input, total_channel_size);
	default:
		break;
	}

	CHECK_CUDA_ERROR(cudaMemcpy(output, gpu_input, size, cudaMemcpyDeviceToHost));

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float gpuElapsedTime = 0;
	cudaEventElapsedTime(&gpuElapsedTime, start, stop);

	cudaFree(gpu_input);
	cudaDeviceReset();
	return gpuElapsedTime;
}