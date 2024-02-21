#include "gamma.cuh"

inline __device__ int gamma_val(float gamma, int val) {
	return static_cast<unsigned char>(__powf(val / 255.0f, gamma) * 255);
}

__device__ unsigned char LUT_device[256];
__constant__ unsigned char LUT_constant[256];

__global__ void k_init_LUT(float gamma) {
	int thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;

	LUT_device[thread_id] = static_cast<unsigned char>(__powf(thread_id / 255.0f, gamma) * 255);
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

__global__ void k_3D_gc_load_balance(unsigned char* input, int rows, int cols, int load) {
	int thread_id = ((blockIdx.x * blockDim.x) + threadIdx.x) * load;

	if (thread_id < rows * cols) {
		for (int i = 0; i < load; i++) {
			input[thread_id + i] = LUT_device[input[thread_id + i]];
		}
	}
}

__global__ void k_3D_gc_vectorized(unsigned char* input, int rows, int cols, int load) {
	int thread_id = ((blockIdx.x * blockDim.x) + threadIdx.x);

	if (load == 4) {
		uchar4 pixel = reinterpret_cast<uchar4*>(input)[thread_id];

		if (thread_id < rows * cols) {
			pixel.x = LUT_device[pixel.x];
			pixel.y = LUT_device[pixel.y];
			pixel.z = LUT_device[pixel.z];
			pixel.w = LUT_device[pixel.w];
		}

		reinterpret_cast<uchar4*>(input)[thread_id] = pixel;
	}
	if (load == 3) {
		uchar3 pixel = reinterpret_cast<uchar3*>(input)[thread_id];

		if (thread_id < rows * cols) {
			pixel.x = LUT_device[pixel.x];
			pixel.y = LUT_device[pixel.y];
			pixel.z = LUT_device[pixel.z];
		}

		reinterpret_cast<uchar3*>(input)[thread_id] = pixel;
	}
	if (load == 2) {
		uchar2 pixel = reinterpret_cast<uchar2*>(input)[thread_id];

		if (thread_id < rows * cols) {
			pixel.x = LUT_device[pixel.x];
			pixel.y = LUT_device[pixel.y];
		}

		reinterpret_cast<uchar2*>(input)[thread_id] = pixel;
	}
}

__global__ void k_3D_gc_vectorized_recompute(unsigned char* input, float gamma, int rows, int cols, int load) {
	int thread_id = ((blockIdx.x * blockDim.x) + threadIdx.x);

	if (load == 4) {
		uchar4 pixel = reinterpret_cast<uchar4*>(input)[thread_id];

		if (thread_id < rows * cols) {
			pixel.x = gamma_val(gamma,pixel.x);
			pixel.y = gamma_val(gamma, pixel.y);
			pixel.z = gamma_val(gamma, pixel.z);
			pixel.w = gamma_val(gamma, pixel.w);
		}

		reinterpret_cast<uchar4*>(input)[thread_id] = pixel;
	}
	if (load == 3) {
		uchar3 pixel = reinterpret_cast<uchar3*>(input)[thread_id];

		if (thread_id < rows * cols) {
			pixel.x = gamma_val(gamma, pixel.x);
			pixel.y = gamma_val(gamma, pixel.y);
			pixel.z = gamma_val(gamma, pixel.z);
		}

		reinterpret_cast<uchar3*>(input)[thread_id] = pixel;
	}
	if (load == 2) {
		uchar2 pixel = reinterpret_cast<uchar2*>(input)[thread_id];

		if (thread_id < rows * cols) {
			pixel.x = gamma_val(gamma, pixel.x);
			pixel.y = gamma_val(gamma, pixel.y);
		}

		reinterpret_cast<uchar2*>(input)[thread_id] = pixel;
	}
}

__global__ void k_3D_gc_vectorized_constant(unsigned char* input, int rows, int cols, int load) {
	int thread_id = ((blockIdx.x * blockDim.x) + threadIdx.x);

	if (load == 4) {
		uchar4 pixel = reinterpret_cast<uchar4*>(input)[thread_id];

		if (thread_id < rows * cols) {
			pixel.x = LUT_constant[pixel.x];
			pixel.y = LUT_constant[pixel.y];
			pixel.z = LUT_constant[pixel.z];
			pixel.w = LUT_constant[pixel.w];
		}

		reinterpret_cast<uchar4*>(input)[thread_id] = pixel;
	}
	if (load == 3) {
		uchar3 pixel = reinterpret_cast<uchar3*>(input)[thread_id];

		if (thread_id < rows * cols) {
			pixel.x = LUT_constant[pixel.x];
			pixel.y = LUT_constant[pixel.y];
			pixel.z = LUT_constant[pixel.z];
		}

		reinterpret_cast<uchar3*>(input)[thread_id] = pixel;
	}
	if (load == 2) {
		uchar2 pixel = reinterpret_cast<uchar2*>(input)[thread_id];

		if (thread_id < rows * cols) {
			pixel.x = LUT_constant[pixel.x];
			pixel.y = LUT_constant[pixel.y];
		}

		reinterpret_cast<uchar2*>(input)[thread_id] = pixel;
	}
}

__global__ void k_3D_gc_vectorized_shared(unsigned char* input, int rows, int cols, int load) {
	int thread_id = ((blockIdx.x * blockDim.x) + threadIdx.x);
	__shared__ unsigned char s_LUT[256];
	int thread_id_in_block = threadIdx.x;

	if (thread_id_in_block < 256) {
		s_LUT[thread_id_in_block] = LUT_device[thread_id_in_block];
	}
	__syncthreads();

	if (load == 4) {
		uchar4 pixel = reinterpret_cast<uchar4*>(input)[thread_id];

		if (thread_id < rows * cols) {
			pixel.x = s_LUT[pixel.x];
			pixel.y = s_LUT[pixel.y];
			pixel.z = s_LUT[pixel.z];
			pixel.w = s_LUT[pixel.w];
		}

		reinterpret_cast<uchar4*>(input)[thread_id] = pixel;
	}
}



float gc_3d_gpu(cv::Mat* output_img, float gamma, GAMMA ver) {
	unsigned char* gpu_input = NULL;

	unsigned int cols = (*output_img).cols * 3;
	unsigned int rows = (*output_img).rows;

	unsigned int pixels = cols * rows;
	unsigned long int size = pixels * sizeof(unsigned char);

	unsigned char* output = output_img->data;

	cudaEvent_t start, stop;
	CHECK_CUDA_ERROR(cudaEventCreate(&start));
	CHECK_CUDA_ERROR(cudaEventCreate(&stop));
	CHECK_CUDA_ERROR(cudaEventRecord(start));

	CHECK_CUDA_ERROR(cudaHostRegister(output, size, 0));
	CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_input, size));
	CHECK_CUDA_ERROR(cudaMemcpy(gpu_input, output, size, cudaMemcpyHostToDevice));

	dim3 block(1024);
	dim3 grid((pixels + block.x - 1) / block.x);

	switch (ver)
	{
	default:
		break;
	case GAMMA_default:
		k_init_LUT << <8, 32 >> > (gamma);
		k_3D_gc << <grid, block >> > (gpu_input, rows, cols);
		break;
	case GAMMA_shared:
		k_init_LUT << <8, 32 >> > (gamma);
		k_3D_gc_shared << <grid, block >> > (gpu_input, rows, cols);
		break;
	case GAMMA_constant:
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
	case GAMMA_loadBalance:
		{
			int load = 4;
			k_init_LUT << <4, 64 >> > (gamma);
			dim3 grid_laod_balance(((size / load) + block.x - 1) / block.x);
			k_3D_gc_load_balance << <grid_laod_balance, block >> > (gpu_input, rows, cols, load);
		}
		break;
	case GAMMA_vectorized:
		{
			int load = 4;
			k_init_LUT << <4, 64 >> > (gamma);
			dim3 grid_laod_balance(((size / load) + block.x - 1) / block.x);
			k_3D_gc_vectorized << <grid_laod_balance, block >> > (gpu_input, rows, cols, load);
		}
		break;
	case GAMMA_vectorized_constant:
		{
			int load = 4;
			dim3 grid_laod_balance(((size / load) + block.x - 1) / block.x);
			unsigned char LUT[256] = { 0 };
			for (int i = 0; i < 256; i++) {
				LUT[i] = static_cast<unsigned char>(pow(i / 255.0f, gamma) * 255);
			}
			CHECK_CUDA_ERROR(cudaMemcpyToSymbol(LUT_constant, LUT, 256 * sizeof(unsigned char)));
			k_3D_gc_vectorized_constant << <grid_laod_balance, block >> > (gpu_input, rows, cols, load);
		}
		break;
	case GAMMA_vectorized_recompute:
		{
			int load = 4;
			dim3 grid_laod_balance(((size / load) + block.x - 1) / block.x);
			k_3D_gc_vectorized_recompute << <grid_laod_balance, block >> > (gpu_input,gamma, rows, cols, load);
		}
		break;
	case GAMMA_vectorized_shared:
		{
			int load = 4;
			dim3 grid_laod_balance(((size / load) + block.x - 1) / block.x);
			k_init_LUT << <4, 64 >> > (gamma);
			k_3D_gc_vectorized_shared << <grid_laod_balance, block >> > (gpu_input, rows, cols, load);
		}
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