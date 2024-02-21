#include "histogram.cuh"

__device__ int dev_histogram[256] = { 0 };
__device__ float dev_normalized_histogram[256] = { 0 };
__device__ float dev_cdf[256] = { 0 };
__device__ unsigned char dev_equalization_values[256] = { 0 };

/*color gpu variables*/

__device__ int dev_histogram_red[256] = { 0 };
__device__ float dev_normalized_histogram_red[256] = { 0 };
__device__ float dev_cdf_red[256] = { 0 };
__device__ unsigned char dev_equalization_values_red[256] = { 0 };

__device__ int dev_histogram_green[256] = { 0 };
__device__ float dev_normalized_histogram_green[256] = { 0 };
__device__ float dev_cdf_green[256] = { 0 };
__device__ unsigned char dev_equalization_values_green[256] = { 0 };

__device__ int dev_histogram_blue[256] = { 0 };
__device__ float dev_normalized_histogram_blue[256] = { 0 };
__device__ float dev_cdf_blue[256] = { 0 };
__device__ unsigned char dev_equalization_values_blue[256] = { 0 };

/*reduce branch variables*/

__device__ int dev_histogram_rb[256 * 3] = { 0 };
__device__ float dev_normalized_histogram_rb[256 * 3] = { 0 };
__device__ float dev_cdf_rb[256 * 3] = { 0 };
__device__ unsigned char dev_equalization_values_rb[256 * 3] = { 0 };

__global__ void k_1D_extract_histogram(unsigned char* input, int pixels) {
	int threadId = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (threadId >= pixels) {
		return;
	}

	atomicAdd(&dev_histogram[input[threadId]], 1);
}

__global__ void k_1D_extract_histogram_load_balance(unsigned char* input, int pixels, int load) {
	int threadId = ((blockIdx.x * blockDim.x) + threadIdx.x) * load;

	if (threadId < pixels) {
		for (int i = 0; i < load; i++) {
			atomicAdd(&dev_histogram[input[threadId + i]], 1);
		}
	}
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

__global__ void k_1D_extract_histogram_vectorized(unsigned char* input, int pixels, int load) {
	int thread_id = ((blockIdx.x * blockDim.x) + threadIdx.x);

	if (thread_id > pixels) {
		return;
	}
	switch (load % 4)
	{
	default:
		break;
	case 0:
		{
			uchar4 pixel = reinterpret_cast<uchar4*>(input)[thread_id];

			atomicAdd(&dev_histogram[pixel.x], 1);
			atomicAdd(&dev_histogram[pixel.y], 1);
			atomicAdd(&dev_histogram[pixel.z], 1);
			atomicAdd(&dev_histogram[pixel.w], 1);
		}
		break;
	case 2:
		{
			uchar2 pixel = reinterpret_cast<uchar2*>(input)[thread_id];

			atomicAdd(&dev_histogram[pixel.x], 1);
			atomicAdd(&dev_histogram[pixel.y], 1);
		}
		break;
	case 3:
		{
			uchar3 pixel = reinterpret_cast<uchar3*>(input)[thread_id];

			atomicAdd(&dev_histogram[pixel.x], 1);
			atomicAdd(&dev_histogram[pixel.y], 1);
			atomicAdd(&dev_histogram[pixel.z], 1);
		}
		break;
	}
}

__global__ void k_1D_extract_histogram_vectorized_shared(unsigned char* input, int pixels, int load) {
	__shared__ int s_histogram[256];

	int thread_id = ((blockIdx.x * blockDim.x) + threadIdx.x);
	int thread_id_in_block = threadIdx.x;

	if (thread_id > pixels) {
		return;
	}

	if (thread_id_in_block < 256) {
		s_histogram[thread_id_in_block] = 0;
	}
	__syncthreads();

	uchar4 pixel = reinterpret_cast<uchar4*>(input)[thread_id];

	atomicAdd(&s_histogram[pixel.x], 1);
	atomicAdd(&s_histogram[pixel.y], 1);
	atomicAdd(&s_histogram[pixel.z], 1);
	atomicAdd(&s_histogram[pixel.w], 1);

	__syncthreads();

	if (thread_id_in_block < 256) {
		atomicAdd(&dev_histogram[thread_id_in_block], s_histogram[thread_id_in_block]);
	}
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

__global__ void k_1D_equalize_load_balance(unsigned char* input, int pixels,int load) {
	int threadId = ((blockIdx.x * blockDim.x) + threadIdx.x) * load;
	if (threadId >= pixels) {
		return;
	}
	for (int i = 0; i < load; i++) {
		input[threadId + i] = static_cast<uchar>(dev_equalization_values[input[threadId + i]]);
	}
}

__global__ void k_1D_equalize_vectorized(unsigned char* input, int pixels, int load) {
	int thread_id = ((blockIdx.x * blockDim.x) + threadIdx.x);

	if (thread_id >= pixels) {
		return;
	}

	switch (load % 4)
	{
	default:
		break;
		case 0:
		{
			uchar4 pixel = reinterpret_cast<uchar4*>(input)[thread_id];

			pixel.x = dev_equalization_values[pixel.x];
			pixel.y = dev_equalization_values[pixel.y];
			pixel.z = dev_equalization_values[pixel.z];
			pixel.w = dev_equalization_values[pixel.w];

			reinterpret_cast<uchar4*>(input)[thread_id] = pixel;
		}
		break;
	case 2:
		{
			uchar2 pixel = reinterpret_cast<uchar2*>(input)[thread_id];

			pixel.x = dev_equalization_values[pixel.x];
			pixel.y = dev_equalization_values[pixel.y];

			reinterpret_cast<uchar2*>(input)[thread_id] = pixel;
		}
		break;
	case 3:
		{
			uchar3 pixel = reinterpret_cast<uchar3*>(input)[thread_id];

			pixel.x = dev_equalization_values[pixel.x];
			pixel.y = dev_equalization_values[pixel.y];
			pixel.z = dev_equalization_values[pixel.z];

			reinterpret_cast<uchar3*>(input)[thread_id] = pixel;
		}
		break;
	}
}

__global__ void k_1D_equalize_vectorized_shared(unsigned char* input, int pixels, int load) {
	__shared__ unsigned char s_equalization_values[256];

	int thread_id = ((blockIdx.x * blockDim.x) + threadIdx.x);
	int thread_id_in_block = threadIdx.x;

	if (thread_id >= pixels) {
		return;
	}

	if (thread_id_in_block < 256) {
		s_equalization_values[thread_id_in_block] = dev_equalization_values[thread_id_in_block];
	}
	__syncthreads();

	uchar4 pixel = reinterpret_cast<uchar4*>(input)[thread_id];

	pixel.x = s_equalization_values[pixel.x];
	pixel.y = s_equalization_values[pixel.y];
	pixel.z = s_equalization_values[pixel.z];
	pixel.w = s_equalization_values[pixel.w];

	reinterpret_cast<uchar4*>(input)[thread_id] = pixel;
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
	__shared__ unsigned char cache_equalization_values[256];

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

__global__ void k_1D_equalize_vectroized_shared(unsigned char* input, int pixels) { /*load the cache before threadId control*/
	__shared__ unsigned char cache_equalization_values[256];

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
	__shared__ unsigned char s_equalization_values[256];

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

float he_1d_gpu(cv::Mat* output_img, HISTOGRAM ver) {
	unsigned char* gpu_input = nullptr;

	unsigned char* output = output_img->data;

	unsigned int cols = (*output_img).cols;
	unsigned int rows = (*output_img).rows;

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
	case HISTOGRAM_load_balance:
		{
			int load = 4;
			dim3 grid_laod_balance(((pixels / load) + block.x - 1) / block.x);
			k_1D_extract_histogram_load_balance << <grid_laod_balance, block >> > (gpu_input, pixels, load);
			k_1D_normalize_cdf_equalization << <4, 64 >> > (pixels);
			k_1D_equalize_load_balance << <grid_laod_balance, block >> > (gpu_input, pixels, load);
		}
		break;
	case HISTOGRAM_vectorized:
		{
			int load = 4;
			dim3 grid_laod_balance(((pixels / load) + block.x - 1) / block.x);
			k_1D_extract_histogram_vectorized << <grid_laod_balance, block >> > (gpu_input, pixels, load);
			k_1D_normalize_cdf_equalization << <4, 64 >> > (pixels);
			k_1D_equalize_vectorized << <grid_laod_balance, block >> > (gpu_input, pixels, load);
		}
		break;
	case HISTOGRAM_vectorized_shared:
		{
			int load = 4;
			dim3 grid_laod_balance(((pixels / load) + block.x - 1) / block.x);
			k_1D_extract_histogram_vectorized_shared << <grid_laod_balance, block >> > (gpu_input, pixels, load);
			k_1D_normalize_cdf_equalization_shared << <4, 64 >> > (pixels);
			k_1D_equalize_vectorized_shared << <grid_laod_balance, block >> > (gpu_input, pixels, load);
		}
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

__global__ void k_3D_extract_histogram_load_balance(unsigned char* input, int pixels, int load) {
	int tid = ((blockIdx.x * blockDim.x) + threadIdx.x) * load;

	if (tid >= pixels) {
		return;
	}

	for (int i = 0; i < load; i++) {
		unsigned char pixel = input[tid + i];
		switch ((tid + i) % 3)
		{
		default:
			break;
		case 0:
			atomicAdd(&dev_histogram_red[pixel], 1);
			break;
		case 1:
			atomicAdd(&dev_histogram_green[pixel], 1);
			break;
		case 2:
			atomicAdd(&dev_histogram_blue[pixel], 1);
			break;
		}
	}
}

__global__ void k_3D_extract_histogram_vectorized(unsigned char* input, int pixels, int load) {
	int tid = ((blockIdx.x * blockDim.x) + threadIdx.x);

	if (tid >= pixels) {
		return;
	}

	switch (load % 4)
	{
	default:
		break;
	case 0:
		{
			uchar4 pixel = reinterpret_cast<uchar4*>(input)[tid];
			switch (tid % 3)
			{
			default:
				break;
			case 0:
				atomicAdd(&dev_histogram_red[pixel.x], 1);
				atomicAdd(&dev_histogram_green[pixel.y], 1);
				atomicAdd(&dev_histogram_blue[pixel.z], 1);
				atomicAdd(&dev_histogram_red[pixel.w], 1);
				break;
			case 1:
				atomicAdd(&dev_histogram_green[pixel.x], 1);
				atomicAdd(&dev_histogram_blue[pixel.y], 1);
				atomicAdd(&dev_histogram_red[pixel.z], 1);
				atomicAdd(&dev_histogram_green[pixel.w], 1);
				break;
			case 2:
				atomicAdd(&dev_histogram_blue[pixel.x], 1);
				atomicAdd(&dev_histogram_red[pixel.y], 1);
				atomicAdd(&dev_histogram_green[pixel.z], 1);
				atomicAdd(&dev_histogram_blue[pixel.w], 1);
				break;
			}
		}
		break;
	case 2:
		{
			uchar2 pixel = reinterpret_cast<uchar2*>(input)[tid];
			switch (tid % 3)
			{
			default:
				break;
			case 0:
				atomicAdd(&dev_histogram_red[pixel.x], 1);
				atomicAdd(&dev_histogram_green[pixel.y], 1);
				break;
			case 1:
				atomicAdd(&dev_histogram_blue[pixel.x], 1);
				atomicAdd(&dev_histogram_red[pixel.y], 1);
				break;
			case 2:
				atomicAdd(&dev_histogram_green[pixel.x], 1);
				atomicAdd(&dev_histogram_blue[pixel.y], 1);
				break;
			}
		}
		break;
	case 3:
		{
			uchar3 pixel = reinterpret_cast<uchar3*>(input)[tid];
			atomicAdd(&dev_histogram_red[pixel.x], 1);
			atomicAdd(&dev_histogram_green[pixel.y], 1);
			atomicAdd(&dev_histogram_blue[pixel.z], 1);
		}
		break;
	}
}

__global__ void k_3D_extract_histogram_vectorized_shared(unsigned char* input, int pixels, int load) {
	int tid = ((blockIdx.x * blockDim.x) + threadIdx.x);
	int tid_in_block = threadIdx.x;
	__shared__ int s_histogram_red[256];
	__shared__ int s_histogram_green[256];
	__shared__ int s_histogram_blue[256];

	if (tid >= pixels) {
		return;
	}

	if (tid_in_block < 256) {
		s_histogram_red[tid_in_block] = 0;
		s_histogram_green[tid_in_block] = 0;
		s_histogram_blue[tid_in_block] = 0;
	}

	switch (load % 4)
	{
	default:
		break;
	case 0:
		{
			uchar4 pixel = reinterpret_cast<uchar4*>(input)[tid];
			switch (tid % 3)
			{
			default:
				break;
			case 0:
				atomicAdd(&s_histogram_red[pixel.x], 1);
				atomicAdd(&s_histogram_green[pixel.y], 1);
				atomicAdd(&s_histogram_blue[pixel.z], 1);
				atomicAdd(&s_histogram_red[pixel.w], 1);
				break;
			case 1:
				atomicAdd(&s_histogram_green[pixel.x], 1);
				atomicAdd(&s_histogram_blue[pixel.y], 1);
				atomicAdd(&s_histogram_red[pixel.z], 1);
				atomicAdd(&s_histogram_green[pixel.w], 1);
				break;
			case 2:
				atomicAdd(&s_histogram_blue[pixel.x], 1);
				atomicAdd(&s_histogram_red[pixel.y], 1);
				atomicAdd(&s_histogram_green[pixel.z], 1);
				atomicAdd(&s_histogram_blue[pixel.w], 1);
				break;
			}
		}
		break;
	}
	__syncthreads();

	if (tid_in_block < 256) {
		atomicAdd(&dev_histogram_red[tid_in_block], s_histogram_red[tid_in_block]);
		atomicAdd(&dev_histogram_green[tid_in_block], s_histogram_green[tid_in_block]);
		atomicAdd(&dev_histogram_blue[tid_in_block], s_histogram_blue[tid_in_block]);
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

__global__ void k_3D_extract_histogram_rb_vectorized(unsigned char* input, int total_channel_size) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (tid < total_channel_size) {
		uchar3 pixel = reinterpret_cast<uchar3*>(input)[tid];

		atomicAdd(&dev_histogram_rb[pixel.x], 1);
		atomicAdd(&dev_histogram_rb[256 + pixel.y], 1);
		atomicAdd(&dev_histogram_rb[512 + pixel.z], 1);
	}
}

__global__ void k_3D_extract_histogram_rb_shared(unsigned char* input, int total_channel_size) {
	__shared__ int s_histogram[256 * 3];
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
	__shared__ int cache_histogram_red[256];
	__shared__ int cache_histogram_green[256];
	__shared__ int cache_histogram_blue[256];

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

	dev_equalization_values_rb[tid] = (dev_cdf_rb[tid] * 255.0f) + 0.5f; /* int donusturmeyi sildim burad*/
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

__global__ void k_3D_equalize_load_balance(unsigned char* input, int pixels, int load) {
	int tid = ((blockIdx.x * blockDim.x) + threadIdx.x) * load;

	if (tid >= pixels) {
		return;
	}

	for (int i = 0; i < load; i++) {
		int pixelid = tid + i;
		unsigned char pixel = input[pixelid];

		switch (pixelid % 3) {
		default:
			break;
		case 0:
			input[pixelid] = static_cast<uchar>(dev_equalization_values_red[pixel]);
			break;
		case 1:
			input[pixelid] = static_cast<uchar>(dev_equalization_values_green[pixel]);
			break;
		case 2:
			input[pixelid] = static_cast<uchar>(dev_equalization_values_blue[pixel]);
			break;
		}
	}
}

__global__ void k_3D_equalize_vectorized(unsigned char* input, int pixels, int load) {
	int tid = ((blockIdx.x * blockDim.x) + threadIdx.x);

	if (tid >= pixels) {
		return;
	}

	switch (load % 4)
	{
	default:
		break;
	case 0:
		{
			uchar4 pixel = reinterpret_cast<uchar4*>(input)[tid];
			switch (tid % 3)
			{
			default:
				break;
			case 0:
				pixel.x = static_cast<uchar>(dev_equalization_values_red[pixel.x]);
				pixel.y = static_cast<uchar>(dev_equalization_values_green[pixel.y]);
				pixel.z = static_cast<uchar>(dev_equalization_values_blue[pixel.z]);
				pixel.w = static_cast<uchar>(dev_equalization_values_red[pixel.w]);
				break;
			case 1:
				pixel.x = static_cast<uchar>(dev_equalization_values_green[pixel.x]);
				pixel.y = static_cast<uchar>(dev_equalization_values_blue[pixel.y]);
				pixel.z = static_cast<uchar>(dev_equalization_values_red[pixel.z]);
				pixel.w = static_cast<uchar>(dev_equalization_values_green[pixel.w]);
				break;
			case 2:
				pixel.x = static_cast<uchar>(dev_equalization_values_blue[pixel.x]);
				pixel.y = static_cast<uchar>(dev_equalization_values_red[pixel.y]);
				pixel.z = static_cast<uchar>(dev_equalization_values_green[pixel.z]);
				pixel.w = static_cast<uchar>(dev_equalization_values_blue[pixel.w]);
				break;
			}
			reinterpret_cast<uchar4*>(input)[tid] = pixel;
		}
		break;
	case 2:
		{
			uchar2 pixel = reinterpret_cast<uchar2*>(input)[tid];
			switch (tid % 3)
			{
			default:
				break;
			case 0:
				pixel.x = static_cast<uchar>(dev_equalization_values_red[pixel.x]);
				pixel.y = static_cast<uchar>(dev_equalization_values_green[pixel.y]);
				break;
			case 1:
				pixel.x = static_cast<uchar>(dev_equalization_values_blue[pixel.x]);
				pixel.y = static_cast<uchar>(dev_equalization_values_red[pixel.y]);
				break;
			case 2:
				pixel.x = static_cast<uchar>(dev_equalization_values_green[pixel.x]);
				pixel.y = static_cast<uchar>(dev_equalization_values_blue[pixel.y]);
				break;
			}
			reinterpret_cast<uchar2*>(input)[tid] = pixel;
		}
		break;
	case 3:
		{
			uchar3 pixel = reinterpret_cast<uchar3*>(input)[tid];

			pixel.x = static_cast<uchar>(dev_equalization_values_red[pixel.x]);
			pixel.y = static_cast<uchar>(dev_equalization_values_green[pixel.y]);
			pixel.z = static_cast<uchar>(dev_equalization_values_blue[pixel.z]);

			reinterpret_cast<uchar3*>(input)[tid] = pixel;
		}
		break;
	}
}

__global__ void k_3D_equalize_vectorized_shared(unsigned char* input, int pixels, int load) {
	int tid = ((blockIdx.x * blockDim.x) + threadIdx.x);
	int tid_in_block = threadIdx.x;

	__shared__ unsigned char s_equalization_values_red[256];
	__shared__ unsigned char s_equalization_values_green[256];
	__shared__ unsigned char s_equalization_values_blue[256];

	if (tid >= pixels) {
		return;
	}

	if (tid_in_block < 256) {
		s_equalization_values_red[tid_in_block] = dev_equalization_values_red[tid_in_block];
		s_equalization_values_green[tid_in_block] = dev_equalization_values_green[tid_in_block];
		s_equalization_values_blue[tid_in_block] = dev_equalization_values_blue[tid_in_block];
	}

	switch (load % 4)
	{
	default:
		break;
	case 0:
		{
			uchar4 pixel = reinterpret_cast<uchar4*>(input)[tid];
			switch (tid % 3)
			{
			default:
				break;
			case 0:
				pixel.x = static_cast<uchar>(s_equalization_values_red[pixel.x]);
				pixel.y = static_cast<uchar>(s_equalization_values_green[pixel.y]);
				pixel.z = static_cast<uchar>(s_equalization_values_blue[pixel.z]);
				pixel.w = static_cast<uchar>(s_equalization_values_red[pixel.w]);
				break;
			case 1:
				pixel.x = static_cast<uchar>(s_equalization_values_green[pixel.x]);
				pixel.y = static_cast<uchar>(s_equalization_values_blue[pixel.y]);
				pixel.z = static_cast<uchar>(s_equalization_values_red[pixel.z]);
				pixel.w = static_cast<uchar>(s_equalization_values_green[pixel.w]);
				break;
			case 2:
				pixel.x = static_cast<uchar>(s_equalization_values_blue[pixel.x]);
				pixel.y = static_cast<uchar>(s_equalization_values_red[pixel.y]);
				pixel.z = static_cast<uchar>(s_equalization_values_green[pixel.z]);
				pixel.w = static_cast<uchar>(s_equalization_values_blue[pixel.w]);
				break;
			}
			reinterpret_cast<uchar4*>(input)[tid] = pixel;
		}
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

__global__ void k_3D_equalize_rb_vectorized(unsigned char* input, int total_channel_size) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (tid >= total_channel_size) {
		return;
	}
	uchar3 pixel = reinterpret_cast<uchar3*>(input)[tid];

	pixel.x = dev_equalization_values_rb[pixel.x];
	pixel.y = dev_equalization_values_rb[256 + pixel.y];
	pixel.z = dev_equalization_values_rb[512 + pixel.z];

	reinterpret_cast<uchar3*>(input)[tid] = pixel;
}

__global__ void k_3D_equalize_shared(unsigned char* input, int total_pixel_size) {
	__shared__ unsigned char cache_equalization_values_red[256];
	__shared__ unsigned char cache_equalization_values_green[256];
	__shared__ unsigned char cache_equalization_values_blue[256];

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

float he_3d_gpu(cv::Mat* output_img, HISTOGRAM ver) {
	unsigned char* gpu_input = nullptr;

	unsigned char* output = output_img->data;

	unsigned int cols = (*output_img).cols;
	unsigned int rows = (*output_img).rows;

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
		break;
	case HISTOGRAM_load_balance:
		{
			int load = 4;
			dim3 grid_laod_balance(((total_channel_size / load) + block.x - 1) / block.x);
			k_3D_extract_histogram_load_balance << <grid_laod_balance, block >> > (gpu_input, total_channel_size, load);
			k_3D_normalize_cdf_equalization << <4, 64 >> > (pixels);
			k_3D_equalize_load_balance << <grid_laod_balance, block >> > (gpu_input, total_channel_size, load);
		}
		break;
	case HISTOGRAM_vectorized:
		{
			int load = 4;
			dim3 grid_laod_balance(((total_channel_size / load) + block.x - 1) / block.x);
			k_3D_extract_histogram_vectorized << <grid_laod_balance, block >> > (gpu_input, total_channel_size, load);
			k_3D_normalize_cdf_equalization << <4, 64 >> > (pixels);
			k_3D_equalize_vectorized << <grid_laod_balance, block >> > (gpu_input, total_channel_size, load);
		}
		break;
	case HISTOGRAM_vectorized_shared:
		{
			int load = 4;
			dim3 grid_laod_balance(((total_channel_size / load) + block.x - 1) / block.x);
			k_3D_extract_histogram_vectorized_shared << <grid_laod_balance, block >> > (gpu_input, total_channel_size, load);
			k_3D_normalize_cdf_equalization_shared << <1, 256 >> > (pixels);
			k_3D_equalize_vectorized_shared << <grid_laod_balance, block >> > (gpu_input, total_channel_size, load);
		}
		break;
	case HISTOGRAM_vectorized_reduce_branches:
		{
			int load = 3;
			dim3 grid_laod_balance(((total_channel_size / load) + block.x - 1) / block.x);
			k_3D_extract_histogram_rb_vectorized << <grid_laod_balance, block >> > (gpu_input, total_channel_size);
			k_3D_normalize_cdf_equalization_rb << <12, 64 >> > (pixels);
			k_3D_equalize_rb_vectorized << <grid_laod_balance, block >> > (gpu_input, total_channel_size);
		}
		break;
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