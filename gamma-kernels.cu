#include "gamma.cuh"

__device__ unsigned char LUT_device[256];
__constant__ unsigned char LUT_constant[256];

__global__ void k_init_LUT(float gamma) {
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int threadId = (blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x);

	LUT_device[threadId] = static_cast<unsigned char>(__powf(threadId / 255.0f, gamma) * 255);
}

__global__ void k_3D_gc(unsigned char* input, int rows, int cols) {
	int thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (thread_id >= rows * cols) {
		return;
	}
	input[thread_id] = LUT_device[input[thread_id]];
}


__global__ void k_3D_gc_shared(unsigned char* input, int rows, int cols) {
	__shared__ unsigned char cache_LUT[256];

	int thread_id_in_block = threadIdx.x;
	int thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (thread_id_in_block < 256) {
		cache_LUT[thread_id_in_block] = LUT_device[thread_id_in_block];
	}

	if (thread_id >= rows * cols) {
		return;
	}
	__syncthreads();

	input[thread_id] = cache_LUT[input[thread_id]];
}

__global__ void k_3D_gc_constant(unsigned char* input, int rows, int cols) {
	int thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (thread_id >= rows * cols) {
		return;
	}
	input[thread_id] = LUT_constant[input[thread_id]];
}

__global__ void k_3D_gc_recompute(unsigned char* input, int rows, int cols, float gamma) {
	__shared__ unsigned char s_LUT[256];
	int thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;
	int thread_id_in_block = threadIdx.x;

	if (thread_id_in_block < 256) {
		s_LUT[thread_id_in_block] = static_cast<unsigned char>(__powf(thread_id_in_block / 255.0f, gamma) * 255);
	}
	__syncthreads();
	if (thread_id < rows * cols) {
		input[thread_id] = s_LUT[input[thread_id]];
	}
}

__global__ void k_3D_gc_noLUT(unsigned char* input, int rows, int cols, float gamma) {
	int thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;
	unsigned char pixel = input[thread_id];
	if (thread_id < rows * cols) {
		input[thread_id] = static_cast<unsigned char>(__powf(pixel / 255.0f, gamma) * 255);
	}
}

__global__ void k_3D_gc_kernel_fusion(unsigned char* input, int rows, int cols, float gamma) {
	int thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (thread_id >= rows * cols) {
		return;
	}
	if (thread_id < 256) {
		LUT_device[thread_id] = static_cast<unsigned char>(__powf(thread_id / 255.0f, gamma) * 255);
	}
	__syncthreads();
	input[thread_id] = LUT_device[input[thread_id]];
}

float gc_3d_gpu(cv::Mat input_img, cv::Mat* output_img, float gamma, GAMMA ver) {
	unsigned char* gpu_input = NULL;

	unsigned int cols = input_img.cols * 3;
	unsigned int rows = input_img.rows;
	unsigned long int size = cols * rows * sizeof(unsigned char);

	unsigned char* output = output_img->data;

	cudaEvent_t start, stop;
	CHECK_CUDA_ERROR(cudaEventCreate(&start));
	CHECK_CUDA_ERROR(cudaEventCreate(&stop));
	CHECK_CUDA_ERROR(cudaEventRecord(start));

	CHECK_CUDA_ERROR(cudaHostRegister(output, size, 0));
	CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_input, size));
	CHECK_CUDA_ERROR(cudaMemcpy(gpu_input, output, size, cudaMemcpyHostToDevice));

	dim3 block(1024);
	dim3 grid((size + block.x - 1) / block.x);

	switch (ver)
	{
	default:
		break;
	case GAMMA_default: /*default ver*/
		k_init_LUT << <8, 32 >> > (gamma);
		k_3D_gc << <grid, block >> > (gpu_input, rows, cols);
		break;
	case GAMMA_shared: /*shared ver*/
		k_init_LUT << <8, 32 >> > (gamma);
		k_3D_gc_shared << <grid, block >> > (gpu_input, rows, cols);
		break;
	case GAMMA_constant: /*constant ver*/
		{
			unsigned char LUT[256] = { 0 };
			for (int i = 0; i < 256; i++) {
				LUT[i] = static_cast<unsigned char>(pow(i / 255.0f, gamma) * 255);
			}
			CHECK_CUDA_ERROR(cudaMemcpyToSymbol(LUT_constant, LUT, 256 * sizeof(unsigned char)));
		}
		k_3D_gc_constant << <grid, block >> > (gpu_input, rows, cols);
		break;
	case GAMMA_recompute: /*recompute*/
		k_3D_gc_recompute << <grid, block >> > (gpu_input, rows, cols, gamma);
		break;
	case GAMMA_noLUT:
		k_3D_gc_noLUT << <grid, block >> > (gpu_input, rows, cols, gamma);
		break;
	}
	cudaDeviceSynchronize();
	CHECK_CUDA_ERROR(cudaGetLastError());

	CHECK_CUDA_ERROR(cudaMemcpy(output, gpu_input, size, cudaMemcpyDeviceToHost));

	CHECK_CUDA_ERROR(cudaEventRecord(stop));
	CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

	float gpuElapsedTime = 0;
	CHECK_CUDA_ERROR(cudaEventElapsedTime(&gpuElapsedTime, start, stop));

	CHECK_CUDA_ERROR(cudaFree(gpu_input));
	CHECK_CUDA_ERROR(cudaHostUnregister(output));

	CHECK_CUDA_ERROR(cudaEventDestroy(start));
	CHECK_CUDA_ERROR(cudaEventDestroy(stop));
	CHECK_CUDA_ERROR(cudaDeviceReset());

	return gpuElapsedTime;
}