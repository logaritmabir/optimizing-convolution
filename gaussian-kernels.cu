#include "gaussian.cuh"

__constant__ unsigned char dev_const_conv_kernel[3][3];

__global__ void k_1D_gf(unsigned char* input, int rows, int cols, int mask_dim)
{
	int ty = blockIdx.x * blockDim.x + threadIdx.x;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;
	int threadId = (tx * cols + ty);

	unsigned char conv_kernel[3][3] = { {1, 2, 1}, {2, 4, 2}, {1, 2, 1} };

	int new_val = 0;
	int offset = 1;

	if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1)) {
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				new_val += conv_kernel[i][j] * input[(tx - offset + i) * cols + ty - offset + j];
			}
		}
		input[threadId] = static_cast<uchar>(new_val / 16);
	}
}

__global__ void k_1D_gf_unroll(unsigned char* input, int rows, int cols, int mask_dim)
{
	int ty = blockIdx.x * blockDim.x + threadIdx.x;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;
	int threadId = (tx * cols + ty);

	unsigned char conv_kernel[3][3] = { {1, 2, 1}, {2, 4, 2}, {1, 2, 1} };

	int new_val = 0;

	if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1)) {
		new_val += conv_kernel[0][0] * input[(tx - 1) * cols + ty - 1];
		new_val += conv_kernel[0][1] * input[(tx - 1) * cols + ty];
		new_val += conv_kernel[0][2] * input[(tx - 1) * cols + ty + 1];
		new_val += conv_kernel[1][0] * input[tx * cols + ty - 1];
		new_val += conv_kernel[1][1] * input[tx * cols + ty];
		new_val += conv_kernel[1][2] * input[tx * cols + ty + 1];
		new_val += conv_kernel[2][0] * input[(tx + 1) * cols + ty - 1];
		new_val += conv_kernel[2][1] * input[(tx + 1) * cols + ty];
		new_val += conv_kernel[2][2] * input[(tx + 1) * cols + ty + 1];

		input[threadId] = static_cast<uchar>(new_val / 16);
	}
}

__global__ void k_1D_gf_prefetch(unsigned char* input, int rows, int cols, int mask_dim)
{
	int ty = blockIdx.x * blockDim.x + threadIdx.x;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;
	int threadId = (tx * cols + ty);

	unsigned char conv_kernel[3][3] = { {1, 2, 1}, {2, 4, 2}, {1, 2, 1} };

	int new_val = 0;
	int offset = 1;

	int x_index = tx - offset;
	int y_index = ty - offset;
	unsigned char pixel = input[x_index * cols + y_index];

	if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1)) { /*conv element prefetch*/
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				new_val += conv_kernel[i][j] * pixel;
				pixel = input[x_index * cols + (++y_index)];
			}
			y_index = ty - offset;
			pixel = input[(++x_index) * cols + y_index];
		}
		input[threadId] = static_cast<uchar>(new_val / 16);
	}
}

__global__ void k_1D_gf_constant(unsigned char* input, int rows, int cols, int mask_dim)
{
	int ty = blockIdx.x * blockDim.x + threadIdx.x;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;
	int threadId = (tx * cols + ty);

	int new_val = 0;
	int offset = 1;

	if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1)) {
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				new_val += dev_const_conv_kernel[i][j] * input[(tx - offset + i) * cols + ty - offset + j];
			}
		}
		input[threadId] = static_cast<uchar>(new_val / 16);
	}
}

__global__ void k_1D_gf_shared(unsigned char* input, int rows, int cols, int mask_dim)
{
	unsigned char conv_kernel[3][3] = { {1, 2, 1}, {2, 4, 2}, {1, 2, 1} };

	__shared__  unsigned char cache[34][36];

	int ty = blockIdx.x * blockDim.x + threadIdx.x;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;
	int threadId = (tx * cols + ty);

	unsigned int bx = threadIdx.y;
	unsigned int by = threadIdx.x;

	unsigned int cy = by + 1;
	unsigned int cx = bx + 1;

	cache[cx][cy] = input[tx * cols + ty];

	if (cx == 1) {
		cache[0][cy] = input[((tx - 1) * cols + ty)];
	}
	if (cx == 32) {
		cache[33][cy] = input[((tx + 1) * cols + ty)];
	}
	if (cy == 1) {
		cache[cx][0] = input[((tx)*cols + ty - 1)];
	}
	if (cy == 32) {
		cache[cx][33] = input[((tx)*cols + ty + 1)];
	}
	__syncthreads();

	int new_val = 0;
	int offset = 1;

	if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1)) {
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				new_val += conv_kernel[i][j] * cache[(cx - offset + i)][cy - offset + j];
			}
		}
	}
	else {
		return;
	}

	input[threadId] = static_cast<uchar>(new_val / 16);
}

__global__ void k_1D_gf_combined(unsigned char* input, int rows, int cols, int mask_dim)
{
	__shared__  unsigned char cache[34][36];

	int ty = blockIdx.x * blockDim.x + threadIdx.x;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;
	int threadId = (tx * cols + ty);

	unsigned int bx = threadIdx.y;
	unsigned int by = threadIdx.x;

	unsigned int cy = by + 1;
	unsigned int cx = bx + 1;

	cache[cx][cy] = input[tx * cols + ty];

	if (cx == 1) {
		cache[0][cy] = input[((tx - 1) * cols + ty)];
	}
	if (cx == 32) {
		cache[33][cy] = input[((tx + 1) * cols + ty)];
	}
	if (cy == 1) {
		cache[cx][0] = input[((tx)*cols + ty - 1)];
	}
	if (cy == 32) {
		cache[cx][33] = input[((tx)*cols + ty + 1)];
	}

	__syncthreads();
	int new_val = 0;

	if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1)) {
		new_val += dev_const_conv_kernel[0][0] * cache[cx - 1][cy - 1];
		new_val += dev_const_conv_kernel[0][1] * cache[cx - 1][cy];
		new_val += dev_const_conv_kernel[0][2] * cache[cx - 1][cy + 1];

		new_val += dev_const_conv_kernel[1][0] * cache[cx][cy - 1];
		new_val += dev_const_conv_kernel[1][1] * cache[cx][cy];
		new_val += dev_const_conv_kernel[1][2] * cache[cx][cy + 1];

		new_val += dev_const_conv_kernel[2][0] * cache[cx + 1][cy - 1];
		new_val += dev_const_conv_kernel[2][1] * cache[cx + 1][cy];
		new_val += dev_const_conv_kernel[2][2] * cache[cx + 1][cy + 1];
	}
	else {
		return;
	}

	input[threadId] = static_cast<uchar>(new_val / 16);
}

float gf_1d_gpu(cv::Mat input_img, cv::Mat* output_img, GAUSSIAN ver)
{
	unsigned char* gpu_input = nullptr;
	unsigned char* output = output_img->data;

	unsigned int cols = input_img.cols;
	unsigned int rows = input_img.rows;
	unsigned int size = cols * rows * sizeof(unsigned char);

	unsigned char conv_kernel[3][3] = { {1, 2, 1}, {2, 4, 2}, {1, 2, 1} };

	const int mask_dim = 3;

	dim3 block(32, 32);
	dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	cudaHostRegister(output, size, 0);

	CHECK_CUDA_ERROR(cudaMalloc((unsigned char**)&gpu_input, size));
	CHECK_CUDA_ERROR(cudaMemcpy(gpu_input, output, size, cudaMemcpyHostToDevice));

	switch (ver)
	{
	default:
		break;
	case GAUSSIAN_default:
		k_1D_gf << <grid, block >> > (gpu_input, rows, cols, mask_dim);
		break;
	case GAUSSIAN_unroll:
		k_1D_gf_unroll << <grid, block >> > (gpu_input, rows, cols, mask_dim);
		break;
	case GAUSSIAN_prefetch:
		k_1D_gf_prefetch << <grid, block >> > (gpu_input, rows, cols, mask_dim);
		break;
	case GAUSSIAN_constant:
		CHECK_CUDA_ERROR(cudaMemcpyToSymbol(dev_const_conv_kernel, conv_kernel, sizeof(uchar) * 3 * 3));
		k_1D_gf_constant << <grid, block >> > (gpu_input, rows, cols, mask_dim);
		break;
	case GAUSSIAN_shared:
		k_1D_gf_shared << <grid, block >> > (gpu_input, rows, cols, mask_dim);
		break;
	case GAUSSIAN_combined:
		CHECK_CUDA_ERROR(cudaMemcpyToSymbol(dev_const_conv_kernel, conv_kernel, sizeof(uchar) * 3 * 3));
		k_1D_gf_combined << <grid, block >> > (gpu_input, rows, cols, mask_dim);
		break;
	}
	CHECK_CUDA_ERROR(cudaMemcpy(output, gpu_input, size, cudaMemcpyDeviceToHost));

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float elapsed = 0.0f;
	cudaEventElapsedTime(&elapsed, start, stop);

	cudaHostUnregister(output);
	cudaFree(gpu_input);
	cudaDeviceReset();
	return elapsed;
}

__global__ void k_3D_gf(unsigned char* input, int rows, int cols, int mask_dim)
{
	unsigned char conv_kernel[3][3] = { {1, 2, 1}, {2, 4, 2}, {1, 2, 1} };

	int ty = blockIdx.x * blockDim.x + threadIdx.x;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;
	int threadId = (tx * cols + ty);

	int new_val = 0;
	int offset_x = 1, offset_y = 3;

	if ((tx > 2 && tx < rows - 2) && (ty > 2 && ty < cols - 2)) {
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				new_val += conv_kernel[i][j] * input[(tx + i - offset_x) * cols + (ty + (j * 3) - offset_y)];
			}
		}
	}
	else {
		return;
	}

	input[threadId] = new_val >> 4;
}

__global__ void k_3D_gf_unroll(unsigned char* input, int rows, int cols, int mask_dim)
{
	unsigned char conv_kernel[3][3] = { {1, 2, 1}, {2, 4, 2}, {1, 2, 1} };

	int ty = blockIdx.x * blockDim.x + threadIdx.x;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;
	int threadId = (tx * cols + ty);

	int new_val = 0;

	if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1)) {
		new_val += conv_kernel[0][0] * input[(tx - 1) * cols + ty - 3];
		new_val += conv_kernel[0][1] * input[(tx - 1) * cols + ty];
		new_val += conv_kernel[0][2] * input[(tx - 1) * cols + ty + 3];
		new_val += conv_kernel[1][0] * input[tx * cols + ty - 3];
		new_val += conv_kernel[1][1] * input[tx * cols + ty];
		new_val += conv_kernel[1][2] * input[tx * cols + ty + 3];
		new_val += conv_kernel[2][0] * input[(tx + 1) * cols + ty - 3];
		new_val += conv_kernel[2][1] * input[(tx + 1) * cols + ty];
		new_val += conv_kernel[2][2] * input[(tx + 1) * cols + ty + 3];
	}
	else {
		return;
	}

	input[threadId] = static_cast<uchar>(new_val / 16);
}

__global__ void k_3D_gf_prefetch(unsigned char* input, int rows, int cols, int mask_dim)
{
	unsigned char conv_kernel[3][3] = { {1, 2, 1}, {2, 4, 2}, {1, 2, 1} };

	int ty = blockIdx.x * blockDim.x + threadIdx.x;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;
	int threadId = (tx * cols + ty);

	int new_val = 0;
	int offset_x = 1, offset_y = 3;

	int x_index = tx - offset_x;
	int y_index = ty - offset_y;
	unsigned char pixel = input[x_index * cols + y_index];

	if ((tx > 2 && tx < rows - 2) && (ty > 2 && ty < cols - 2)) {
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				new_val += conv_kernel[i][j] * pixel;
				y_index += 3;
				pixel = input[x_index * cols + y_index];
			}
			y_index = ty - offset_y;
			pixel = input[(++x_index) * cols + y_index];
		}
	}
	else {
		return;
	}
	input[threadId] = static_cast<uchar>(new_val / 16);
}


__global__ void k_3D_gf_constant(unsigned char* input, int rows, int cols, int mask_dim)
{
	int ty = blockIdx.x * blockDim.x + threadIdx.x;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;
	int threadId = (tx * cols + ty);

	int new_val = 0;
	int offset_x = 1, offset_y = 3;

	if ((tx > 2 && tx < rows - 2) && (ty > 2 && ty < cols - 2)) {
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				new_val += dev_const_conv_kernel[i][j] * input[(tx + i - offset_x) * cols + (ty + (j * 3) - offset_y)];
			}
		}
	}
	else {
		return;
	}

	//if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1)) {
	//	new_val += dev_const_conv_kernel[0][0] * input[(tx - 1) * cols + ty - 3];
	//	new_val += dev_const_conv_kernel[0][1] * input[(tx - 1) * cols + ty];
	//	new_val += dev_const_conv_kernel[0][2] * input[(tx - 1) * cols + ty + 3];
	//	new_val += dev_const_conv_kernel[1][0] * input[tx * cols + ty - 3];
	//	new_val += dev_const_conv_kernel[1][1] * input[tx * cols + ty];
	//	new_val += dev_const_conv_kernel[1][2] * input[tx * cols + ty + 3];
	//	new_val += dev_const_conv_kernel[2][0] * input[(tx + 1) * cols + ty - 3];
	//	new_val += dev_const_conv_kernel[2][1] * input[(tx + 1) * cols + ty];
	//	new_val += dev_const_conv_kernel[2][2] * input[(tx + 1) * cols + ty + 3];
	//}
	//else {
	//	return;
	//}
	input[threadId] = static_cast<uchar>(new_val / 16);
}

__global__ void k_3D_gf_shared(unsigned char* input, int rows, int cols, int mask_dim)
{
	unsigned char conv_kernel[3][3] = { {1, 2, 1}, {2, 4, 2}, {1, 2, 1} };

	__shared__ unsigned char cache[34][38];

	int ty = blockIdx.x * blockDim.x + threadIdx.x;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;
	int threadId = (tx * cols + ty);

	unsigned int bx = threadIdx.y;
	unsigned int by = threadIdx.x;

	unsigned int cy = by + 3;
	unsigned int cx = bx + 1;

	cache[cx][cy] = input[threadId];

	if (cx == 1) {
		cache[0][cy] = input[((tx - 1) * cols + ty)];
	}
	if (cx == 32) {
		cache[33][cy] = input[((tx + 1) * cols + ty)];
	}
	if (cy == 3) {
		cache[cx][0] = input[(tx * cols + ty - 3)];
		cache[cx][1] = input[(tx * cols + ty - 2)];
		cache[cx][2] = input[(tx * cols + ty - 1)];
	}
	if (cy == 34) {
		cache[cx][35] = input[(tx * cols + ty + 1)];
		cache[cx][36] = input[(tx * cols + ty + 2)];
		cache[cx][37] = input[(tx * cols + ty + 3)];
	}

	__syncthreads();

	int new_val = 0;
	int offset_x = 1, offset_y = 3;

	if ((tx > 2 && tx < rows - 2) && (ty > 2 && ty < cols - 2)) {
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				new_val += conv_kernel[i][j] * cache[cx + i - offset_x][cy + (j * 3) - offset_y];
			}
		}
	}
	else {
		return;
	}

	//if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1)) {
	//	new_val += conv_kernel[0][0] * cache[cx - 1][cy - 3];
	//	new_val += conv_kernel[0][1] * cache[cx - 1][cy];
	//	new_val += conv_kernel[0][2] * cache[cx - 1][cy + 3];

	//	new_val += conv_kernel[1][0] * cache[cx][cy - 3];
	//	new_val += conv_kernel[1][1] * cache[cx][cy];
	//	new_val += conv_kernel[1][2] * cache[cx][cy + 3];

	//	new_val += conv_kernel[2][0] * cache[cx + 1][cy - 3];
	//	new_val += conv_kernel[2][1] * cache[cx + 1][cy];
	//	new_val += conv_kernel[2][2] * cache[cx + 1][cy + 3];
	//}
	//else {
	//	return;
	//}

	input[threadId] = static_cast<uchar>(new_val / 16);
}

__global__ void k_3D_gf_combined(unsigned char* input, int rows, int cols, int mask_dim)
{
	__shared__ unsigned char cache[34][38];

	int ty = blockIdx.x * blockDim.x + threadIdx.x;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;
	int threadId = (tx * cols + ty);

	unsigned int bx = threadIdx.y;
	unsigned int by = threadIdx.x;

	unsigned int cy = by + 3;
	unsigned int cx = bx + 1;

	cache[cx][cy] = input[threadId];

	if (cx == 1) {
		cache[0][cy] = input[((tx - 1) * cols + ty)];
	}
	if (cx == 32) {
		cache[33][cy] = input[((tx + 1) * cols + ty)];
	}
	if (cy == 3) {
		cache[cx][0] = input[(tx * cols + ty - 3)];
		cache[cx][1] = input[(tx * cols + ty - 2)];
		cache[cx][2] = input[(tx * cols + ty - 1)];
	}
	if (cy == 34) {
		cache[cx][35] = input[(tx * cols + ty + 1)];
		cache[cx][36] = input[(tx * cols + ty + 2)];
		cache[cx][37] = input[(tx * cols + ty + 3)];
	}

	__syncthreads();
	int new_val = 0;
	int offset_x = 1, offset_y = 3;

	if ((tx > 2 && tx < rows - 2) && (ty > 2 && ty < cols - 2)) {
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				new_val += dev_const_conv_kernel[i][j] * cache[cx + i - offset_x][cy + (j * 3) - offset_y];
			}
		}
	}
	else {
		return;
	}

	//if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1)) {
	//	new_val += dev_const_conv_kernel[0][0] * cache[cx - 1][cy - 3];
	//	new_val += dev_const_conv_kernel[0][1] * cache[cx - 1][cy];
	//	new_val += dev_const_conv_kernel[0][2] * cache[cx - 1][cy + 3];

	//	new_val += dev_const_conv_kernel[1][0] * cache[cx][cy - 3];
	//	new_val += dev_const_conv_kernel[1][1] * cache[cx][cy];
	//	new_val += dev_const_conv_kernel[1][2] * cache[cx][cy + 3];

	//	new_val += dev_const_conv_kernel[2][0] * cache[cx + 1][cy - 3];
	//	new_val += dev_const_conv_kernel[2][1] * cache[cx + 1][cy];
	//	new_val += dev_const_conv_kernel[2][2] * cache[cx + 1][cy + 3];
	//}
	//else {
	//	return;
	//}

	input[threadId] = static_cast<uchar>(new_val / 16);
}

float gf_3d_gpu(cv::Mat input_img, cv::Mat* output_img, GAUSSIAN ver)
{
	unsigned char* gpu_input = NULL;
	unsigned char* output = output_img->data;

	unsigned int cols = input_img.cols * 3;
	unsigned int rows = input_img.rows;
	unsigned int size = rows * cols * sizeof(unsigned char);

	unsigned char conv_kernel[3][3] = { {1, 2, 1}, {2, 4, 2}, {1, 2, 1} };

	const uint mask_dim = 3;

	dim3 block(32, 32);
	dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	cudaHostRegister(output, size, 0);

	CHECK_CUDA_ERROR(cudaMalloc((unsigned char**)&gpu_input, size));
	CHECK_CUDA_ERROR(cudaMemcpy(gpu_input, output, size, cudaMemcpyHostToDevice));

	switch (ver)
	{
	default:
		break;
	case GAUSSIAN_default:
		k_3D_gf << <grid, block >> > (gpu_input, rows, cols, mask_dim);
		break;
	case GAUSSIAN_prefetch:
		k_3D_gf_prefetch << <grid, block >> > (gpu_input, rows, cols, mask_dim);
		break;
	case GAUSSIAN_unroll:
		k_3D_gf_unroll << <grid, block >> > (gpu_input, rows, cols, mask_dim);
		break;
	case GAUSSIAN_constant:
		CHECK_CUDA_ERROR(cudaMemcpyToSymbol(dev_const_conv_kernel, conv_kernel, sizeof(uchar) * 3 * 3));
		k_3D_gf_constant << <grid, block >> > (gpu_input, rows, cols, mask_dim);
		break;
	case GAUSSIAN_shared:
		k_3D_gf_shared << <grid, block >> > (gpu_input, rows, cols, mask_dim);
		break;
	case GAUSSIAN_combined:
		CHECK_CUDA_ERROR(cudaMemcpyToSymbol(dev_const_conv_kernel, conv_kernel, sizeof(uchar) * 3 * 3));
		k_3D_gf_combined << <grid, block >> > (gpu_input, rows, cols, mask_dim);
		break;
	}

	CHECK_CUDA_ERROR(cudaMemcpy(output, gpu_input, size, cudaMemcpyDeviceToHost));
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float elapsed = 0.0f;
	cudaEventElapsedTime(&elapsed, start, stop);

	cudaFree(gpu_input);
	cudaDeviceReset();
	return elapsed;
}
