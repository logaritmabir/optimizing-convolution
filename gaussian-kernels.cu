#include "gaussian.cuh"

__constant__ unsigned char const_conv_kernel3x3[3][3];

__device__ unsigned char global_conv_kernel3x3[3][3] = {{1, 2, 1}, 
														{2, 4, 2}, 
														{1, 2, 1} };

__global__ void k_1D_gf_3x3_global(unsigned char* input, int rows, int cols)
{
	int ty = blockIdx.x * blockDim.x + threadIdx.x;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	int new_val = 0;

	if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1)) {
		new_val += global_conv_kernel3x3[0][0] * input[(tx - 1) * cols + ty - 1];
		new_val += global_conv_kernel3x3[0][1] * input[(tx - 1) * cols + ty];
		new_val += global_conv_kernel3x3[0][2] * input[(tx - 1) * cols + ty + 1];
		new_val += global_conv_kernel3x3[1][0] * input[tx * cols + ty - 1];
		new_val += global_conv_kernel3x3[1][1] * input[tx * cols + ty];
		new_val += global_conv_kernel3x3[1][2] * input[tx * cols + ty + 1];
		new_val += global_conv_kernel3x3[2][0] * input[(tx + 1) * cols + ty - 1];
		new_val += global_conv_kernel3x3[2][1] * input[(tx + 1) * cols + ty];
		new_val += global_conv_kernel3x3[2][2] * input[(tx + 1) * cols + ty + 1];

		input[tx * cols + ty] = new_val >> 4;
	}
}

__global__ void k_1D_gf_3x3_local(unsigned char* input, int rows, int cols)
{
	const int ty = blockIdx.x * blockDim.x + threadIdx.x;
	const int tx = blockIdx.y * blockDim.y + threadIdx.y;

	unsigned char conv_kernel3x3[3][3] = { {1, 2, 1},
											{2, 4, 2},
											{1, 2, 1} };
	int new_val = 0;

	if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1)) {
		new_val += conv_kernel3x3[0][0] * input[(tx - 1) * cols + ty - 1];
		new_val += conv_kernel3x3[0][1] * input[(tx - 1) * cols + ty];
		new_val += conv_kernel3x3[0][2] * input[(tx - 1) * cols + ty + 1];
		new_val += conv_kernel3x3[1][0] * input[tx * cols + ty - 1];
		new_val += conv_kernel3x3[1][1] * input[tx * cols + ty];
		new_val += conv_kernel3x3[1][2] * input[tx * cols + ty + 1];
		new_val += conv_kernel3x3[2][0] * input[(tx + 1) * cols + ty - 1];
		new_val += conv_kernel3x3[2][1] * input[(tx + 1) * cols + ty];
		new_val += conv_kernel3x3[2][2] * input[(tx + 1) * cols + ty + 1];

		input[tx * cols + ty] = new_val >> 4;
	}
}

__global__ void k_1D_gf_3x3_constant(unsigned char* input, int rows, int cols)
{
	int ty = blockIdx.x * blockDim.x + threadIdx.x;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	int new_val = 0;

	if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1)) {
		new_val += const_conv_kernel3x3[0][0] * input[(tx - 1) * cols + ty - 1];
		new_val += const_conv_kernel3x3[0][1] * input[(tx - 1) * cols + ty];
		new_val += const_conv_kernel3x3[0][2] * input[(tx - 1) * cols + ty + 1];
		new_val += const_conv_kernel3x3[1][0] * input[tx * cols + ty - 1];
		new_val += const_conv_kernel3x3[1][1] * input[tx * cols + ty];
		new_val += const_conv_kernel3x3[1][2] * input[tx * cols + ty + 1];
		new_val += const_conv_kernel3x3[2][0] * input[(tx + 1) * cols + ty - 1];
		new_val += const_conv_kernel3x3[2][1] * input[(tx + 1) * cols + ty];
		new_val += const_conv_kernel3x3[2][2] * input[(tx + 1) * cols + ty + 1];

		input[tx * cols + ty] = new_val >> 4;
	}
}

__global__ void k_1D_gf_3x3_load_balance32_global(unsigned char* input, int rows, int cols)
{
	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 32;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	for (int i = 0; i < 32; i++) {
		int _ty = ty + i;
		int new_val = 0;

		if ((tx > 0 && tx < rows - 1) && (_ty > 0 && _ty < cols - 1)) {
			new_val += global_conv_kernel3x3[0][0] * input[(tx - 1) * cols + _ty - 1];
			new_val += global_conv_kernel3x3[0][1] * input[(tx - 1) * cols + _ty];
			new_val += global_conv_kernel3x3[0][2] * input[(tx - 1) * cols + _ty + 1];
			new_val += global_conv_kernel3x3[1][0] * input[tx * cols + _ty - 1];
			new_val += global_conv_kernel3x3[1][1] * input[tx * cols + _ty];
			new_val += global_conv_kernel3x3[1][2] * input[tx * cols + _ty + 1];
			new_val += global_conv_kernel3x3[2][0] * input[(tx + 1) * cols + _ty - 1];
			new_val += global_conv_kernel3x3[2][1] * input[(tx + 1) * cols + _ty];
			new_val += global_conv_kernel3x3[2][2] * input[(tx + 1) * cols + _ty + 1];

			input[(tx * cols + _ty)] = new_val >> 4;
		}
	}
}

__global__ void k_1D_gf_3x3_load_balance16_global(unsigned char* input, int rows, int cols)
{
	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 16;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	for (int i = 0; i < 16; i++) {
		int _ty = ty + i;
		int new_val = 0;

		if ((tx > 0 && tx < rows - 1) && (_ty > 0 && _ty < cols - 1)) {
			new_val += global_conv_kernel3x3[0][0] * input[(tx - 1) * cols + _ty - 1];
			new_val += global_conv_kernel3x3[0][1] * input[(tx - 1) * cols + _ty];
			new_val += global_conv_kernel3x3[0][2] * input[(tx - 1) * cols + _ty + 1];
			new_val += global_conv_kernel3x3[1][0] * input[tx * cols + _ty - 1];
			new_val += global_conv_kernel3x3[1][1] * input[tx * cols + _ty];
			new_val += global_conv_kernel3x3[1][2] * input[tx * cols + _ty + 1];
			new_val += global_conv_kernel3x3[2][0] * input[(tx + 1) * cols + _ty - 1];
			new_val += global_conv_kernel3x3[2][1] * input[(tx + 1) * cols + _ty];
			new_val += global_conv_kernel3x3[2][2] * input[(tx + 1) * cols + _ty + 1];

			input[(tx * cols + _ty)] = new_val >> 4;
		}
	}
}

__global__ void k_1D_gf_3x3_load_balance12_global(unsigned char* input, int rows, int cols)
{
	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 12;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	for (int i = 0; i < 12; i++) {
		int _ty = ty + i;
		int new_val = 0;

		if ((tx > 0 && tx < rows - 1) && (_ty > 0 && _ty < cols - 1)) {
			new_val += global_conv_kernel3x3[0][0] * input[(tx - 1) * cols + _ty - 1];
			new_val += global_conv_kernel3x3[0][1] * input[(tx - 1) * cols + _ty];
			new_val += global_conv_kernel3x3[0][2] * input[(tx - 1) * cols + _ty + 1];
			new_val += global_conv_kernel3x3[1][0] * input[tx * cols + _ty - 1];
			new_val += global_conv_kernel3x3[1][1] * input[tx * cols + _ty];
			new_val += global_conv_kernel3x3[1][2] * input[tx * cols + _ty + 1];
			new_val += global_conv_kernel3x3[2][0] * input[(tx + 1) * cols + _ty - 1];
			new_val += global_conv_kernel3x3[2][1] * input[(tx + 1) * cols + _ty];
			new_val += global_conv_kernel3x3[2][2] * input[(tx + 1) * cols + _ty + 1];

			input[(tx * cols + _ty)] = new_val >> 4;
		}
	}
}

__global__ void k_1D_gf_3x3_load_balance8_global(unsigned char* input, int rows, int cols)
{
	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	for (int i = 0; i < 8; i++) {
		int _ty = ty + i;
		int new_val = 0;

		if ((tx > 0 && tx < rows - 1) && (_ty > 0 && _ty < cols - 1)) {
			new_val += global_conv_kernel3x3[0][0] * input[(tx - 1) * cols + _ty - 1];
			new_val += global_conv_kernel3x3[0][1] * input[(tx - 1) * cols + _ty];
			new_val += global_conv_kernel3x3[0][2] * input[(tx - 1) * cols + _ty + 1];
			new_val += global_conv_kernel3x3[1][0] * input[tx * cols + _ty - 1];
			new_val += global_conv_kernel3x3[1][1] * input[tx * cols + _ty];
			new_val += global_conv_kernel3x3[1][2] * input[tx * cols + _ty + 1];
			new_val += global_conv_kernel3x3[2][0] * input[(tx + 1) * cols + _ty - 1];
			new_val += global_conv_kernel3x3[2][1] * input[(tx + 1) * cols + _ty];
			new_val += global_conv_kernel3x3[2][2] * input[(tx + 1) * cols + _ty + 1];

			input[(tx * cols + _ty)] = new_val >> 4;
		}
	}
}

__global__ void k_1D_gf_3x3_load_balance4_global(unsigned char* input, int rows, int cols)
{
	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	for (int i = 0; i < 4; i++) {
		int _ty = ty + i;
		int new_val = 0;

		if ((tx > 0 && tx < rows - 1) && (_ty > 0 && _ty < cols - 1)) {
			new_val += global_conv_kernel3x3[0][0] * input[(tx - 1) * cols + _ty - 1];
			new_val += global_conv_kernel3x3[0][1] * input[(tx - 1) * cols + _ty];
			new_val += global_conv_kernel3x3[0][2] * input[(tx - 1) * cols + _ty + 1];
			new_val += global_conv_kernel3x3[1][0] * input[tx * cols + _ty - 1];
			new_val += global_conv_kernel3x3[1][1] * input[tx * cols + _ty];
			new_val += global_conv_kernel3x3[1][2] * input[tx * cols + _ty + 1];
			new_val += global_conv_kernel3x3[2][0] * input[(tx + 1) * cols + _ty - 1];
			new_val += global_conv_kernel3x3[2][1] * input[(tx + 1) * cols + _ty];
			new_val += global_conv_kernel3x3[2][2] * input[(tx + 1) * cols + _ty + 1];

			input[(tx * cols + _ty)] = new_val >> 4;
		}
	}
}

__global__ void k_1D_gf_3x3_load_balance2_global(unsigned char* input, int rows, int cols)
{
	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	for (int i = 0; i < 2; i++) {
		int _ty = ty + i;
		int new_val = 0;

		if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1)) {
			new_val += global_conv_kernel3x3[0][0] * input[(tx - 1) * cols + _ty - 1];
			new_val += global_conv_kernel3x3[0][1] * input[(tx - 1) * cols + _ty];
			new_val += global_conv_kernel3x3[0][2] * input[(tx - 1) * cols + _ty + 1];
			new_val += global_conv_kernel3x3[1][0] * input[tx * cols + _ty - 1];
			new_val += global_conv_kernel3x3[1][1] * input[tx * cols + _ty];
			new_val += global_conv_kernel3x3[1][2] * input[tx * cols + _ty + 1];
			new_val += global_conv_kernel3x3[2][0] * input[(tx + 1) * cols + _ty - 1];
			new_val += global_conv_kernel3x3[2][1] * input[(tx + 1) * cols + _ty];
			new_val += global_conv_kernel3x3[2][2] * input[(tx + 1) * cols + _ty + 1];

			input[(tx * cols + _ty)] = new_val >> 4;
		}
	}
}

__global__ void k_1D_gf_3x3_load_balance16_local(unsigned char* input, int rows, int cols)
{
	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 16;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	unsigned char conv_kernel[3][3] = { {1, 2, 1}, {2, 4, 2}, {1, 2, 1} };

	for (int i = 0; i < 16; i++) {
		int _ty = ty + i;
		int new_val = 0;

		if ((tx > 0 && tx < rows - 1) && (_ty > 0 && _ty < cols - 1)) {
			new_val += conv_kernel[0][0] * input[(tx - 1) * cols + _ty - 1];
			new_val += conv_kernel[0][1] * input[(tx - 1) * cols + _ty];
			new_val += conv_kernel[0][2] * input[(tx - 1) * cols + _ty + 1];
			new_val += conv_kernel[1][0] * input[tx * cols + _ty - 1];
			new_val += conv_kernel[1][1] * input[tx * cols + _ty];
			new_val += conv_kernel[1][2] * input[tx * cols + _ty + 1];
			new_val += conv_kernel[2][0] * input[(tx + 1) * cols + _ty - 1];
			new_val += conv_kernel[2][1] * input[(tx + 1) * cols + _ty];
			new_val += conv_kernel[2][2] * input[(tx + 1) * cols + _ty + 1];

			input[(tx * cols + _ty)] = new_val >> 4;
		}
	}
}

__global__ void k_1D_gf_3x3_load_balance12_local(unsigned char* input, int rows, int cols)
{
	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 12;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	unsigned char conv_kernel[3][3] = { {1, 2, 1}, {2, 4, 2}, {1, 2, 1} };

	for (int i = 0; i < 12; i++) {
		int _ty = ty + i;
		int new_val = 0;

		if ((tx > 0 && tx < rows - 1) && (_ty > 0 && _ty < cols - 1)) {
			new_val += conv_kernel[0][0] * input[(tx - 1) * cols + _ty - 1];
			new_val += conv_kernel[0][1] * input[(tx - 1) * cols + _ty];
			new_val += conv_kernel[0][2] * input[(tx - 1) * cols + _ty + 1];
			new_val += conv_kernel[1][0] * input[tx * cols + _ty - 1];
			new_val += conv_kernel[1][1] * input[tx * cols + _ty];
			new_val += conv_kernel[1][2] * input[tx * cols + _ty + 1];
			new_val += conv_kernel[2][0] * input[(tx + 1) * cols + _ty - 1];
			new_val += conv_kernel[2][1] * input[(tx + 1) * cols + _ty];
			new_val += conv_kernel[2][2] * input[(tx + 1) * cols + _ty + 1];

			input[(tx * cols + _ty)] = new_val >> 4;
		}
	}
}

__global__ void k_1D_gf_3x3_load_balance8_local(unsigned char* input, int rows, int cols)
{
	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	unsigned char conv_kernel[3][3] = { {1, 2, 1}, {2, 4, 2}, {1, 2, 1} };

	for (int i = 0; i < 8; i++) {
		int _ty = ty + i;
		int new_val = 0;

		if ((tx > 0 && tx < rows - 1) && (_ty > 0 && _ty < cols - 1)) {
			new_val += conv_kernel[0][0] * input[(tx - 1) * cols + _ty - 1];
			new_val += conv_kernel[0][1] * input[(tx - 1) * cols + _ty];
			new_val += conv_kernel[0][2] * input[(tx - 1) * cols + _ty + 1];
			new_val += conv_kernel[1][0] * input[tx * cols + _ty - 1];
			new_val += conv_kernel[1][1] * input[tx * cols + _ty];
			new_val += conv_kernel[1][2] * input[tx * cols + _ty + 1];
			new_val += conv_kernel[2][0] * input[(tx + 1) * cols + _ty - 1];
			new_val += conv_kernel[2][1] * input[(tx + 1) * cols + _ty];
			new_val += conv_kernel[2][2] * input[(tx + 1) * cols + _ty + 1];

			input[(tx * cols + _ty)] = new_val >> 4;
		}
	}
}

__global__ void k_1D_gf_3x3_load_balance4_local(unsigned char* input, int rows, int cols)
{
	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	unsigned char conv_kernel[3][3] = { {1, 2, 1}, {2, 4, 2}, {1, 2, 1} };

	for (int i = 0; i < 4; i++) {
		int _ty = ty + i;
		int new_val = 0;

		if ((tx > 0 && tx < rows - 1) && (_ty > 0 && _ty < cols - 1)) {
			new_val += conv_kernel[0][0] * input[(tx - 1) * cols + _ty - 1];
			new_val += conv_kernel[0][1] * input[(tx - 1) * cols + _ty];
			new_val += conv_kernel[0][2] * input[(tx - 1) * cols + _ty + 1];
			new_val += conv_kernel[1][0] * input[tx * cols + _ty - 1];
			new_val += conv_kernel[1][1] * input[tx * cols + _ty];
			new_val += conv_kernel[1][2] * input[tx * cols + _ty + 1];
			new_val += conv_kernel[2][0] * input[(tx + 1) * cols + _ty - 1];
			new_val += conv_kernel[2][1] * input[(tx + 1) * cols + _ty];
			new_val += conv_kernel[2][2] * input[(tx + 1) * cols + _ty + 1];

			input[(tx * cols + _ty)] = new_val >> 4;
		}
	}
}

__global__ void k_1D_gf_3x3_load_balance2_local(unsigned char* input, int rows, int cols)
{
	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	unsigned char conv_kernel[3][3] = { {1, 2, 1}, {2, 4, 2}, {1, 2, 1} };

	for (int i = 0; i < 2; i++) {
		int _ty = ty + i;
		int new_val = 0;

		if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1)) {
			new_val += conv_kernel[0][0] * input[(tx - 1) * cols + _ty - 1];
			new_val += conv_kernel[0][1] * input[(tx - 1) * cols + _ty];
			new_val += conv_kernel[0][2] * input[(tx - 1) * cols + _ty + 1];
			new_val += conv_kernel[1][0] * input[tx * cols + _ty - 1];
			new_val += conv_kernel[1][1] * input[tx * cols + _ty];
			new_val += conv_kernel[1][2] * input[tx * cols + _ty + 1];
			new_val += conv_kernel[2][0] * input[(tx + 1) * cols + _ty - 1];
			new_val += conv_kernel[2][1] * input[(tx + 1) * cols + _ty];
			new_val += conv_kernel[2][2] * input[(tx + 1) * cols + _ty + 1];

			input[(tx * cols + _ty)] = new_val >> 4;
		}
	}
}

__global__ void k_1D_gf_3x3_load_balance16_constant(unsigned char* input, int rows, int cols)
{
	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 16;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	for (int i = 0; i < 16; i++) {
		int _ty = ty + i;
		int new_val = 0;

		if ((tx > 0 && tx < rows - 1) && (_ty > 0 && _ty < cols - 1)) {
			new_val += const_conv_kernel3x3[0][0] * input[(tx - 1) * cols + _ty - 1];
			new_val += const_conv_kernel3x3[0][1] * input[(tx - 1) * cols + _ty];
			new_val += const_conv_kernel3x3[0][2] * input[(tx - 1) * cols + _ty + 1];
			new_val += const_conv_kernel3x3[1][0] * input[tx * cols + _ty - 1];
			new_val += const_conv_kernel3x3[1][1] * input[tx * cols + _ty];
			new_val += const_conv_kernel3x3[1][2] * input[tx * cols + _ty + 1];
			new_val += const_conv_kernel3x3[2][0] * input[(tx + 1) * cols + _ty - 1];
			new_val += const_conv_kernel3x3[2][1] * input[(tx + 1) * cols + _ty];
			new_val += const_conv_kernel3x3[2][2] * input[(tx + 1) * cols + _ty + 1];

			input[(tx * cols + _ty)] = new_val >> 4;
		}
	}
}

__global__ void k_1D_gf_3x3_load_balance12_constant(unsigned char* input, int rows, int cols)
{
	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 12;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	for (int i = 0; i < 12; i++) {
		int _ty = ty + i;
		int new_val = 0;

		if ((tx > 0 && tx < rows - 1) && (_ty > 0 && _ty < cols - 1)) {
			new_val += const_conv_kernel3x3[0][0] * input[(tx - 1) * cols + _ty - 1];
			new_val += const_conv_kernel3x3[0][1] * input[(tx - 1) * cols + _ty];
			new_val += const_conv_kernel3x3[0][2] * input[(tx - 1) * cols + _ty + 1];
			new_val += const_conv_kernel3x3[1][0] * input[tx * cols + _ty - 1];
			new_val += const_conv_kernel3x3[1][1] * input[tx * cols + _ty];
			new_val += const_conv_kernel3x3[1][2] * input[tx * cols + _ty + 1];
			new_val += const_conv_kernel3x3[2][0] * input[(tx + 1) * cols + _ty - 1];
			new_val += const_conv_kernel3x3[2][1] * input[(tx + 1) * cols + _ty];
			new_val += const_conv_kernel3x3[2][2] * input[(tx + 1) * cols + _ty + 1];

			input[(tx * cols + _ty)] = new_val >> 4;
		}
	}
}

__global__ void k_1D_gf_3x3_load_balance8_constant(unsigned char* input, int rows, int cols)
{
	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	for (int i = 0; i < 8; i++) {
		int _ty = ty + i;
		int new_val = 0;

		if ((tx > 0 && tx < rows - 1) && (_ty > 0 && _ty < cols - 1)) {
			new_val += const_conv_kernel3x3[0][0] * input[(tx - 1) * cols + _ty - 1];
			new_val += const_conv_kernel3x3[0][1] * input[(tx - 1) * cols + _ty];
			new_val += const_conv_kernel3x3[0][2] * input[(tx - 1) * cols + _ty + 1];
			new_val += const_conv_kernel3x3[1][0] * input[tx * cols + _ty - 1];
			new_val += const_conv_kernel3x3[1][1] * input[tx * cols + _ty];
			new_val += const_conv_kernel3x3[1][2] * input[tx * cols + _ty + 1];
			new_val += const_conv_kernel3x3[2][0] * input[(tx + 1) * cols + _ty - 1];
			new_val += const_conv_kernel3x3[2][1] * input[(tx + 1) * cols + _ty];
			new_val += const_conv_kernel3x3[2][2] * input[(tx + 1) * cols + _ty + 1];

			input[(tx * cols + _ty)] = new_val >> 4;
		}
	}
}

__global__ void k_1D_gf_3x3_load_balance4_constant(unsigned char* input, int rows, int cols)
{
	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	for (int i = 0; i < 4; i++) {
		int _ty = ty + i;
		int new_val = 0;

		if ((tx > 0 && tx < rows - 1) && (_ty > 0 && _ty < cols - 1)) {
			new_val += const_conv_kernel3x3[0][0] * input[(tx - 1) * cols + _ty - 1];
			new_val += const_conv_kernel3x3[0][1] * input[(tx - 1) * cols + _ty];
			new_val += const_conv_kernel3x3[0][2] * input[(tx - 1) * cols + _ty + 1];
			new_val += const_conv_kernel3x3[1][0] * input[tx * cols + _ty - 1];
			new_val += const_conv_kernel3x3[1][1] * input[tx * cols + _ty];
			new_val += const_conv_kernel3x3[1][2] * input[tx * cols + _ty + 1];
			new_val += const_conv_kernel3x3[2][0] * input[(tx + 1) * cols + _ty - 1];
			new_val += const_conv_kernel3x3[2][1] * input[(tx + 1) * cols + _ty];
			new_val += const_conv_kernel3x3[2][2] * input[(tx + 1) * cols + _ty + 1];

			input[(tx * cols + _ty)] = new_val >> 4;
		}
	}
}

__global__ void k_1D_gf_3x3_load_balance2_constant(unsigned char* input, int rows, int cols)
{
	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	for (int i = 0; i < 2; i++) {
		int _ty = ty + i;
		int new_val = 0;

		if ((tx > 0 && tx < rows - 1) && (_ty > 0 && _ty < cols - 1)) {
			new_val += const_conv_kernel3x3[0][0] * input[(tx - 1) * cols + _ty - 1];
			new_val += const_conv_kernel3x3[0][1] * input[(tx - 1) * cols + _ty];
			new_val += const_conv_kernel3x3[0][2] * input[(tx - 1) * cols + _ty + 1];
			new_val += const_conv_kernel3x3[1][0] * input[tx * cols + _ty - 1];
			new_val += const_conv_kernel3x3[1][1] * input[tx * cols + _ty];
			new_val += const_conv_kernel3x3[1][2] * input[tx * cols + _ty + 1];
			new_val += const_conv_kernel3x3[2][0] * input[(tx + 1) * cols + _ty - 1];
			new_val += const_conv_kernel3x3[2][1] * input[(tx + 1) * cols + _ty];
			new_val += const_conv_kernel3x3[2][2] * input[(tx + 1) * cols + _ty + 1];

			input[(tx * cols + _ty)] = new_val >> 4;
		}
	}
}

__global__ void k_1D_gf_3x3_vectorized32_global(unsigned char* input, int rows, int cols)
{
	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 32;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	uchar3 top, bot, mid;
	int vals[32] = { 0 };

	if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty + 31 < cols - 1)) {
		for (int i = 0; i < 32; i++) {
			int _ty = ty + i;

			top = reinterpret_cast<uchar3*>(&input[((tx - 1) * cols + _ty - 1)])[0];
			mid = reinterpret_cast<uchar3*>(&input[((tx)*cols + _ty - 1)])[0];
			bot = reinterpret_cast<uchar3*>(&input[((tx + 1) * cols + _ty - 1)])[0];

			if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1)) {
				vals[i] += global_conv_kernel3x3[0][0] * top.x;
				vals[i] += global_conv_kernel3x3[0][1] * top.y;
				vals[i] += global_conv_kernel3x3[0][2] * top.z;
				vals[i] += global_conv_kernel3x3[1][0] * mid.x;
				vals[i] += global_conv_kernel3x3[1][1] * mid.y;
				vals[i] += global_conv_kernel3x3[1][2] * mid.z;
				vals[i] += global_conv_kernel3x3[2][0] * bot.x;
				vals[i] += global_conv_kernel3x3[2][1] * bot.y;
				vals[i] += global_conv_kernel3x3[2][2] * bot.z;
			}
		}
		reinterpret_cast<uchar4*>(&input[(tx * cols + ty)])[0] = make_uchar4(vals[0] >> 4, vals[1] >> 4, vals[2] >> 4, vals[3] >> 4);
		reinterpret_cast<uchar4*>(&input[(tx * cols + ty + 4)])[0] = make_uchar4(vals[4] >> 4, vals[5] >> 4, vals[6] >> 4, vals[7] >> 4);
		reinterpret_cast<uchar4*>(&input[(tx * cols + ty + 8)])[0] = make_uchar4(vals[8] >> 4, vals[9] >> 4, vals[10] >> 4, vals[11] >> 4);
		reinterpret_cast<uchar4*>(&input[(tx * cols + ty + 12)])[0] = make_uchar4(vals[12] >> 4, vals[13] >> 4, vals[14] >> 4, vals[15] >> 4);
		reinterpret_cast<uchar4*>(&input[(tx * cols + ty + 16)])[0] = make_uchar4(vals[16] >> 4, vals[17] >> 4, vals[18] >> 4, vals[19] >> 4);
		reinterpret_cast<uchar4*>(&input[(tx * cols + ty + 20)])[0] = make_uchar4(vals[20] >> 4, vals[21] >> 4, vals[22] >> 4, vals[23] >> 4);
		reinterpret_cast<uchar4*>(&input[(tx * cols + ty + 24)])[0] = make_uchar4(vals[24] >> 4, vals[25] >> 4, vals[26] >> 4, vals[27] >> 4);
		reinterpret_cast<uchar4*>(&input[(tx * cols + ty + 28)])[0] = make_uchar4(vals[28] >> 4, vals[29] >> 4, vals[30] >> 4, vals[31] >> 4);
	}
}


__global__ void k_1D_gf_3x3_vectorized16_global(unsigned char* input, int rows, int cols)
{
	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 16;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	uchar3 top, bot, mid;
	int vals[16] = { 0 };

	if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty + 15 < cols - 1)) {
		for (int i = 0; i < 16; i++) {
			int _ty = ty + i;

			top = reinterpret_cast<uchar3*>(&input[((tx - 1) * cols + _ty - 1)])[0];
			mid = reinterpret_cast<uchar3*>(&input[((tx)*cols + _ty - 1)])[0];
			bot = reinterpret_cast<uchar3*>(&input[((tx + 1) * cols + _ty - 1)])[0];

			if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1)) {
				vals[i] += global_conv_kernel3x3[0][0] * top.x;
				vals[i] += global_conv_kernel3x3[0][1] * top.y;
				vals[i] += global_conv_kernel3x3[0][2] * top.z;
				vals[i] += global_conv_kernel3x3[1][0] * mid.x;
				vals[i] += global_conv_kernel3x3[1][1] * mid.y;
				vals[i] += global_conv_kernel3x3[1][2] * mid.z;
				vals[i] += global_conv_kernel3x3[2][0] * bot.x;
				vals[i] += global_conv_kernel3x3[2][1] * bot.y;
				vals[i] += global_conv_kernel3x3[2][2] * bot.z;
			}
		}
		reinterpret_cast<uchar4*>(&input[(tx * cols + ty)])[0] = make_uchar4(vals[0] >> 4, vals[1] >> 4, vals[2] >> 4, vals[3] >> 4);
		reinterpret_cast<uchar4*>(&input[(tx * cols + ty + 4)])[0] = make_uchar4(vals[4] >> 4, vals[5] >> 4, vals[6] >> 4, vals[7] >> 4);
		reinterpret_cast<uchar4*>(&input[(tx * cols + ty + 8)])[0] = make_uchar4(vals[8] >> 4, vals[9] >> 4, vals[10] >> 4, vals[11] >> 4);
		reinterpret_cast<uchar4*>(&input[(tx * cols + ty + 12)])[0] = make_uchar4(vals[12] >> 4, vals[13] >> 4, vals[14] >> 4, vals[15] >> 4);
	}
}

__global__ void k_1D_gf_3x3_vectorized12_global(unsigned char* input, int rows, int cols)
{
	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 12;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	uchar3 top, bot, mid;
	int vals[12] = { 0 };

	if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty + 11 < cols - 1)) {
		for (int i = 0; i < 12; i++) {
			int _ty = ty + i;

			top = reinterpret_cast<uchar3*>(&input[((tx - 1) * cols + _ty - 1)])[0];
			mid = reinterpret_cast<uchar3*>(&input[((tx)*cols + _ty - 1)])[0];
			bot = reinterpret_cast<uchar3*>(&input[((tx + 1) * cols + _ty - 1)])[0];

			if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1)) {
				vals[i] += global_conv_kernel3x3[0][0] * top.x;
				vals[i] += global_conv_kernel3x3[0][1] * top.y;
				vals[i] += global_conv_kernel3x3[0][2] * top.z;
				vals[i] += global_conv_kernel3x3[1][0] * mid.x;
				vals[i] += global_conv_kernel3x3[1][1] * mid.y;
				vals[i] += global_conv_kernel3x3[1][2] * mid.z;
				vals[i] += global_conv_kernel3x3[2][0] * bot.x;
				vals[i] += global_conv_kernel3x3[2][1] * bot.y;
				vals[i] += global_conv_kernel3x3[2][2] * bot.z;
			}
		}
		reinterpret_cast<uchar4*>(&input[(tx * cols + ty)])[0] = make_uchar4(vals[0] >> 4, vals[1] >> 4, vals[2] >> 4, vals[3] >> 4);
		reinterpret_cast<uchar4*>(&input[(tx * cols + ty + 4)])[0] = make_uchar4(vals[4] >> 4, vals[5] >> 4, vals[6] >> 4, vals[7] >> 4);
		reinterpret_cast<uchar4*>(&input[(tx * cols + ty + 8)])[0] = make_uchar4(vals[8] >> 4, vals[9] >> 4, vals[10] >> 4, vals[11] >> 4);
	}
}
__global__ void k_1D_gf_3x3_vectorized8_global(unsigned char* input, int rows, int cols)
{
	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	uchar3 top, bot, mid;
	int vals[8] = { 0 };

	if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty + 7 < cols - 1)) {
		for (int i = 0; i < 8; i++) {
			int _ty = ty + i;

			top = reinterpret_cast<uchar3*>(&input[((tx - 1) * cols + _ty - 1)])[0];
			mid = reinterpret_cast<uchar3*>(&input[((tx)*cols + _ty - 1)])[0];
			bot = reinterpret_cast<uchar3*>(&input[((tx + 1) * cols + _ty - 1)])[0];

			if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1)) {
				vals[i] += global_conv_kernel3x3[0][0] * top.x;
				vals[i] += global_conv_kernel3x3[0][1] * top.y;
				vals[i] += global_conv_kernel3x3[0][2] * top.z;
				vals[i] += global_conv_kernel3x3[1][0] * mid.x;
				vals[i] += global_conv_kernel3x3[1][1] * mid.y;
				vals[i] += global_conv_kernel3x3[1][2] * mid.z;
				vals[i] += global_conv_kernel3x3[2][0] * bot.x;
				vals[i] += global_conv_kernel3x3[2][1] * bot.y;
				vals[i] += global_conv_kernel3x3[2][2] * bot.z;
			}
		}
		reinterpret_cast<uchar4*>(&input[(tx * cols + ty)])[0] = make_uchar4(vals[0] >> 4, vals[1] >> 4, vals[2] >> 4, vals[3] >> 4);
		reinterpret_cast<uchar4*>(&input[(tx * cols + ty + 4)])[0] = make_uchar4(vals[4] >> 4, vals[5] >> 4, vals[6] >> 4, vals[7] >> 4);
	}
}

__global__ void k_1D_gf_3x3_vectorized4_global(unsigned char* input, int rows, int cols)
{
	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	uchar3 top, bot, mid;
	int vals[4] = { 0 };

	if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty + 3 < cols - 1)) {
		for (int i = 0; i < 4; i++) {
			int _ty = ty + i;

			top = reinterpret_cast<uchar3*>(&input[((tx - 1) * cols + _ty - 1)])[0];
			mid = reinterpret_cast<uchar3*>(&input[((tx)*cols + _ty - 1)])[0];
			bot = reinterpret_cast<uchar3*>(&input[((tx + 1) * cols + _ty - 1)])[0];

			if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1)) {
				vals[i] += global_conv_kernel3x3[0][0] * top.x;
				vals[i] += global_conv_kernel3x3[0][1] * top.y;
				vals[i] += global_conv_kernel3x3[0][2] * top.z;
				vals[i] += global_conv_kernel3x3[1][0] * mid.x;
				vals[i] += global_conv_kernel3x3[1][1] * mid.y;
				vals[i] += global_conv_kernel3x3[1][2] * mid.z;
				vals[i] += global_conv_kernel3x3[2][0] * bot.x;
				vals[i] += global_conv_kernel3x3[2][1] * bot.y;
				vals[i] += global_conv_kernel3x3[2][2] * bot.z;
			}
		}
		reinterpret_cast<uchar4*>(&input[(tx * cols + ty)])[0] = make_uchar4(vals[0] >> 4, vals[1] >> 4, vals[2] >> 4, vals[3] >> 4);
	}
}

__global__ void k_1D_gf_3x3_vectorized2_global(unsigned char* input, int rows, int cols)
{
	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	uchar3 top, bot, mid;
	int vals[2] = { 0 };

	if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty + 1 < cols - 1)) {
		for (int i = 0; i < 2; i++) {
			int _ty = ty + i;

			top = reinterpret_cast<uchar3*>(&input[((tx - 1) * cols + _ty - 1)])[0];
			mid = reinterpret_cast<uchar3*>(&input[((tx)*cols + _ty - 1)])[0];
			bot = reinterpret_cast<uchar3*>(&input[((tx + 1) * cols + _ty - 1)])[0];

			if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1)) {
				vals[i] += global_conv_kernel3x3[0][0] * top.x;
				vals[i] += global_conv_kernel3x3[0][1] * top.y;
				vals[i] += global_conv_kernel3x3[0][2] * top.z;
				vals[i] += global_conv_kernel3x3[1][0] * mid.x;
				vals[i] += global_conv_kernel3x3[1][1] * mid.y;
				vals[i] += global_conv_kernel3x3[1][2] * mid.z;
				vals[i] += global_conv_kernel3x3[2][0] * bot.x;
				vals[i] += global_conv_kernel3x3[2][1] * bot.y;
				vals[i] += global_conv_kernel3x3[2][2] * bot.z;
			}
		}
		reinterpret_cast<uchar2*>(&input[(tx * cols + ty)])[0] = make_uchar2(vals[0] >> 4, vals[1] >> 4);
	}
}
__global__ void k_1D_gf_3x3_vectorized16_local(unsigned char* input, int rows, int cols)
{
	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 16;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	unsigned char conv_kernel3x3[3][3] = { {1, 2, 1}, {2, 4, 2}, {1, 2, 1} };
	uchar3 top, bot, mid;
	int vals[16] = { 0 };

	if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty + 15 < cols - 1)) {
		for (int i = 0; i < 16; i++) {
			int _ty = ty + i;

			top = reinterpret_cast<uchar3*>(&input[((tx - 1) * cols + _ty - 1)])[0];
			mid = reinterpret_cast<uchar3*>(&input[((tx)*cols + _ty - 1)])[0];
			bot = reinterpret_cast<uchar3*>(&input[((tx + 1) * cols + _ty - 1)])[0];

			if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1)) {
				vals[i] += conv_kernel3x3[0][0] * top.x;
				vals[i] += conv_kernel3x3[0][1] * top.y;
				vals[i] += conv_kernel3x3[0][2] * top.z;
				vals[i] += conv_kernel3x3[1][0] * mid.x;
				vals[i] += conv_kernel3x3[1][1] * mid.y;
				vals[i] += conv_kernel3x3[1][2] * mid.z;
				vals[i] += conv_kernel3x3[2][0] * bot.x;
				vals[i] += conv_kernel3x3[2][1] * bot.y;
				vals[i] += conv_kernel3x3[2][2] * bot.z;
			}
		}
		reinterpret_cast<uchar4*>(&input[(tx * cols + ty)])[0] = make_uchar4(vals[0] >> 4, vals[1] >> 4, vals[2] >> 4, vals[3] >> 4);
		reinterpret_cast<uchar4*>(&input[(tx * cols + ty + 4)])[0] = make_uchar4(vals[4] >> 4, vals[5] >> 4, vals[6] >> 4, vals[7] >> 4);
		reinterpret_cast<uchar4*>(&input[(tx * cols + ty + 8)])[0] = make_uchar4(vals[8] >> 4, vals[9] >> 4, vals[10] >> 4, vals[11] >> 4);
		reinterpret_cast<uchar4*>(&input[(tx * cols + ty + 12)])[0] = make_uchar4(vals[12] >> 4, vals[13] >> 4, vals[14] >> 4, vals[15] >> 4);
	}
}

__global__ void k_1D_gf_3x3_vectorized12_local(unsigned char* input, int rows, int cols)
{
	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 12;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	unsigned char conv_kernel3x3[3][3] = { {1, 2, 1}, {2, 4, 2}, {1, 2, 1} };
	uchar3 top, bot, mid;
	int vals[12] = { 0 };

	if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty + 11 < cols - 1)) {
		for (int i = 0; i < 12; i++) {
			int _ty = ty + i;

			top = reinterpret_cast<uchar3*>(&input[((tx - 1) * cols + _ty - 1)])[0];
			mid = reinterpret_cast<uchar3*>(&input[((tx)*cols + _ty - 1)])[0];
			bot = reinterpret_cast<uchar3*>(&input[((tx + 1) * cols + _ty - 1)])[0];

			if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1)) {
				vals[i] += conv_kernel3x3[0][0] * top.x;
				vals[i] += conv_kernel3x3[0][1] * top.y;
				vals[i] += conv_kernel3x3[0][2] * top.z;
				vals[i] += conv_kernel3x3[1][0] * mid.x;
				vals[i] += conv_kernel3x3[1][1] * mid.y;
				vals[i] += conv_kernel3x3[1][2] * mid.z;
				vals[i] += conv_kernel3x3[2][0] * bot.x;
				vals[i] += conv_kernel3x3[2][1] * bot.y;
				vals[i] += conv_kernel3x3[2][2] * bot.z;
			}
		}
		reinterpret_cast<uchar4*>(&input[(tx * cols + ty)])[0] = make_uchar4(vals[0] >> 4, vals[1] >> 4, vals[2] >> 4, vals[3] >> 4);
		reinterpret_cast<uchar4*>(&input[(tx * cols + ty + 4)])[0] = make_uchar4(vals[4] >> 4, vals[5] >> 4, vals[6] >> 4, vals[7] >> 4);
		reinterpret_cast<uchar4*>(&input[(tx * cols + ty + 8)])[0] = make_uchar4(vals[8] >> 4, vals[9] >> 4, vals[10] >> 4, vals[11] >> 4);
	}
}
__global__ void k_1D_gf_3x3_vectorized8_local(unsigned char* input, int rows, int cols)
{
	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	unsigned char conv_kernel3x3[3][3] = { {1, 2, 1}, {2, 4, 2}, {1, 2, 1} };
	uchar3 top, bot, mid;
	int vals[8] = { 0 };

	if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty + 7 < cols - 1)) {
		for (int i = 0; i < 8; i++) {
			int _ty = ty + i;

			top = reinterpret_cast<uchar3*>(&input[((tx - 1) * cols + _ty - 1)])[0];
			mid = reinterpret_cast<uchar3*>(&input[((tx)*cols + _ty - 1)])[0];
			bot = reinterpret_cast<uchar3*>(&input[((tx + 1) * cols + _ty - 1)])[0];

			if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1)) {
				vals[i] += conv_kernel3x3[0][0] * top.x;
				vals[i] += conv_kernel3x3[0][1] * top.y;
				vals[i] += conv_kernel3x3[0][2] * top.z;
				vals[i] += conv_kernel3x3[1][0] * mid.x;
				vals[i] += conv_kernel3x3[1][1] * mid.y;
				vals[i] += conv_kernel3x3[1][2] * mid.z;
				vals[i] += conv_kernel3x3[2][0] * bot.x;
				vals[i] += conv_kernel3x3[2][1] * bot.y;
				vals[i] += conv_kernel3x3[2][2] * bot.z;
			}
		}
		reinterpret_cast<uchar4*>(&input[(tx * cols + ty)])[0] = make_uchar4(vals[0] >> 4, vals[1] >> 4, vals[2] >> 4, vals[3] >> 4);
		reinterpret_cast<uchar4*>(&input[(tx * cols + ty + 4)])[0] = make_uchar4(vals[4] >> 4, vals[5] >> 4, vals[6] >> 4, vals[7] >> 4);
	}
}

__global__ void k_1D_gf_3x3_vectorized4_local(unsigned char* input, int rows, int cols)
{
	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	unsigned char conv_kernel3x3[3][3] = { {1, 2, 1}, {2, 4, 2}, {1, 2, 1} };
	uchar3 top, bot, mid;
	int vals[4] = { 0 };

	if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty + 3 < cols - 1)) {
		for (int i = 0; i < 4; i++) {
			int _ty = ty + i;

			top = reinterpret_cast<uchar3*>(&input[((tx - 1) * cols + _ty - 1)])[0];
			mid = reinterpret_cast<uchar3*>(&input[((tx)*cols + _ty - 1)])[0];
			bot = reinterpret_cast<uchar3*>(&input[((tx + 1) * cols + _ty - 1)])[0];

			if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1)) {
				vals[i] += conv_kernel3x3[0][0] * top.x;
				vals[i] += conv_kernel3x3[0][1] * top.y;
				vals[i] += conv_kernel3x3[0][2] * top.z;
				vals[i] += conv_kernel3x3[1][0] * mid.x;
				vals[i] += conv_kernel3x3[1][1] * mid.y;
				vals[i] += conv_kernel3x3[1][2] * mid.z;
				vals[i] += conv_kernel3x3[2][0] * bot.x;
				vals[i] += conv_kernel3x3[2][1] * bot.y;
				vals[i] += conv_kernel3x3[2][2] * bot.z;
			}
		}
		reinterpret_cast<uchar4*>(&input[(tx * cols + ty)])[0] = make_uchar4(vals[0] >> 4, vals[1] >> 4, vals[2] >> 4, vals[3] >> 4);
	}
}

__global__ void k_1D_gf_3x3_vectorized2_local(unsigned char* input, int rows, int cols)
{
	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	unsigned char conv_kernel3x3[3][3] = { {1, 2, 1}, {2, 4, 2}, {1, 2, 1} };

	uchar3 top, bot, mid;
	int vals[2] = { 0 };

	if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty + 1 < cols - 1)) {
		for (int i = 0; i < 2; i++) {
			int _ty = ty + i;

			top = reinterpret_cast<uchar3*>(&input[((tx - 1) * cols + _ty - 1)])[0];
			mid = reinterpret_cast<uchar3*>(&input[((tx)*cols + _ty - 1)])[0];
			bot = reinterpret_cast<uchar3*>(&input[((tx + 1) * cols + _ty - 1)])[0];

			if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1)) {
				vals[i] += conv_kernel3x3[0][0] * top.x;
				vals[i] += conv_kernel3x3[0][1] * top.y;
				vals[i] += conv_kernel3x3[0][2] * top.z;
				vals[i] += conv_kernel3x3[1][0] * mid.x;
				vals[i] += conv_kernel3x3[1][1] * mid.y;
				vals[i] += conv_kernel3x3[1][2] * mid.z;
				vals[i] += conv_kernel3x3[2][0] * bot.x;
				vals[i] += conv_kernel3x3[2][1] * bot.y;
				vals[i] += conv_kernel3x3[2][2] * bot.z;
			}
		}
		reinterpret_cast<uchar2*>(&input[(tx * cols + ty)])[0] = make_uchar2(vals[0] >> 4, vals[1] >> 4);
	}
}

__global__ void k_1D_gf_3x3_vectorized16_constant(unsigned char* input, int rows, int cols)
{
	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 16;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	uchar3 top, bot, mid;
	int vals[16] = { 0 };

	if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty + 15 < cols - 1)) {
		for (int i = 0; i < 16; i++) {
			int _ty = ty + i;

			top = reinterpret_cast<uchar3*>(&input[((tx - 1) * cols + _ty - 1)])[0];
			mid = reinterpret_cast<uchar3*>(&input[((tx)*cols + _ty - 1)])[0];
			bot = reinterpret_cast<uchar3*>(&input[((tx + 1) * cols + _ty - 1)])[0];

			if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1)) {
				vals[i] += const_conv_kernel3x3[0][0] * top.x;
				vals[i] += const_conv_kernel3x3[0][1] * top.y;
				vals[i] += const_conv_kernel3x3[0][2] * top.z;
				vals[i] += const_conv_kernel3x3[1][0] * mid.x;
				vals[i] += const_conv_kernel3x3[1][1] * mid.y;
				vals[i] += const_conv_kernel3x3[1][2] * mid.z;
				vals[i] += const_conv_kernel3x3[2][0] * bot.x;
				vals[i] += const_conv_kernel3x3[2][1] * bot.y;
				vals[i] += const_conv_kernel3x3[2][2] * bot.z;
			}
		}
		reinterpret_cast<uchar4*>(&input[(tx * cols + ty)])[0] = make_uchar4(vals[0] >> 4, vals[1] >> 4, vals[2] >> 4, vals[3] >> 4);
		reinterpret_cast<uchar4*>(&input[(tx * cols + ty + 4)])[0] = make_uchar4(vals[4] >> 4, vals[5] >> 4, vals[6] >> 4, vals[7] >> 4);
		reinterpret_cast<uchar4*>(&input[(tx * cols + ty + 8)])[0] = make_uchar4(vals[8] >> 4, vals[9] >> 4, vals[10] >> 4, vals[11] >> 4);
		reinterpret_cast<uchar4*>(&input[(tx * cols + ty + 12)])[0] = make_uchar4(vals[12] >> 4, vals[13] >> 4, vals[14] >> 4, vals[15] >> 4);
	}
}

__global__ void k_1D_gf_3x3_vectorized12_constant(unsigned char* input, int rows, int cols)
{
	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 12;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	uchar3 top, bot, mid;
	int vals[12] = { 0 };

	if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty + 11 < cols - 1)) {
		for (int i = 0; i < 12; i++) {
			int _ty = ty + i;

			top = reinterpret_cast<uchar3*>(&input[((tx - 1) * cols + _ty - 1)])[0];
			mid = reinterpret_cast<uchar3*>(&input[((tx)*cols + _ty - 1)])[0];
			bot = reinterpret_cast<uchar3*>(&input[((tx + 1) * cols + _ty - 1)])[0];

			if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1)) {
				vals[i] += const_conv_kernel3x3[0][0] * top.x;
				vals[i] += const_conv_kernel3x3[0][1] * top.y;
				vals[i] += const_conv_kernel3x3[0][2] * top.z;
				vals[i] += const_conv_kernel3x3[1][0] * mid.x;
				vals[i] += const_conv_kernel3x3[1][1] * mid.y;
				vals[i] += const_conv_kernel3x3[1][2] * mid.z;
				vals[i] += const_conv_kernel3x3[2][0] * bot.x;
				vals[i] += const_conv_kernel3x3[2][1] * bot.y;
				vals[i] += const_conv_kernel3x3[2][2] * bot.z;
			}
		}
		reinterpret_cast<uchar4*>(&input[(tx * cols + ty)])[0] = make_uchar4(vals[0] >> 4, vals[1] >> 4, vals[2] >> 4, vals[3] >> 4);
		reinterpret_cast<uchar4*>(&input[(tx * cols + ty + 4)])[0] = make_uchar4(vals[4] >> 4, vals[5] >> 4, vals[6] >> 4, vals[7] >> 4);
		reinterpret_cast<uchar4*>(&input[(tx * cols + ty + 8)])[0] = make_uchar4(vals[8] >> 4, vals[9] >> 4, vals[10] >> 4, vals[11] >> 4);
	}
}
__global__ void k_1D_gf_3x3_vectorized8_constant(unsigned char* input, int rows, int cols)
{
	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	uchar3 top, bot, mid;
	int vals[8] = { 0 };

	if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty + 7 < cols - 1)) {
		for (int i = 0; i < 8; i++) {
			int _ty = ty + i;

			top = reinterpret_cast<uchar3*>(&input[((tx - 1) * cols + _ty - 1)])[0];
			mid = reinterpret_cast<uchar3*>(&input[((tx)*cols + _ty - 1)])[0];
			bot = reinterpret_cast<uchar3*>(&input[((tx + 1) * cols + _ty - 1)])[0];

			if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1)) {
				vals[i] += const_conv_kernel3x3[0][0] * top.x;
				vals[i] += const_conv_kernel3x3[0][1] * top.y;
				vals[i] += const_conv_kernel3x3[0][2] * top.z;
				vals[i] += const_conv_kernel3x3[1][0] * mid.x;
				vals[i] += const_conv_kernel3x3[1][1] * mid.y;
				vals[i] += const_conv_kernel3x3[1][2] * mid.z;
				vals[i] += const_conv_kernel3x3[2][0] * bot.x;
				vals[i] += const_conv_kernel3x3[2][1] * bot.y;
				vals[i] += const_conv_kernel3x3[2][2] * bot.z;
			}
		}
		reinterpret_cast<uchar4*>(&input[(tx * cols + ty)])[0] = make_uchar4(vals[0] >> 4, vals[1] >> 4, vals[2] >> 4, vals[3] >> 4);
		reinterpret_cast<uchar4*>(&input[(tx * cols + ty + 4)])[0] = make_uchar4(vals[4] >> 4, vals[5] >> 4, vals[6] >> 4, vals[7] >> 4);
	}
}

__global__ void k_1D_gf_3x3_vectorized4_constant(unsigned char* input, int rows, int cols)
{
	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	uchar3 top, bot, mid;
	int vals[4] = { 0 };

	if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty + 3 < cols - 1)) {
		for (int i = 0; i < 4; i++) {
			int _ty = ty + i;

			top = reinterpret_cast<uchar3*>(&input[((tx - 1) * cols + _ty - 1)])[0];
			mid = reinterpret_cast<uchar3*>(&input[((tx)*cols + _ty - 1)])[0];
			bot = reinterpret_cast<uchar3*>(&input[((tx + 1) * cols + _ty - 1)])[0];

			if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1)) {
				vals[i] += const_conv_kernel3x3[0][0] * top.x;
				vals[i] += const_conv_kernel3x3[0][1] * top.y;
				vals[i] += const_conv_kernel3x3[0][2] * top.z;
				vals[i] += const_conv_kernel3x3[1][0] * mid.x;
				vals[i] += const_conv_kernel3x3[1][1] * mid.y;
				vals[i] += const_conv_kernel3x3[1][2] * mid.z;
				vals[i] += const_conv_kernel3x3[2][0] * bot.x;
				vals[i] += const_conv_kernel3x3[2][1] * bot.y;
				vals[i] += const_conv_kernel3x3[2][2] * bot.z;
			}
		}
		reinterpret_cast<uchar4*>(&input[(tx * cols + ty)])[0] = make_uchar4(vals[0] >> 4, vals[1] >> 4, vals[2] >> 4, vals[3] >> 4);
	}
}

__global__ void k_1D_gf_3x3_vectorized2_constant(unsigned char* input, int rows, int cols)
{
	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	uchar3 top, bot, mid;
	int vals[2] = { 0 };

	if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty + 1 < cols - 1)) {
		for (int i = 0; i < 2; i++) {
			int _ty = ty + i;

			top = reinterpret_cast<uchar3*>(&input[((tx - 1) * cols + _ty - 1)])[0];
			mid = reinterpret_cast<uchar3*>(&input[((tx)*cols + _ty - 1)])[0];
			bot = reinterpret_cast<uchar3*>(&input[((tx + 1) * cols + _ty - 1)])[0];

			if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1)) {
				vals[i] += const_conv_kernel3x3[0][0] * top.x;
				vals[i] += const_conv_kernel3x3[0][1] * top.y;
				vals[i] += const_conv_kernel3x3[0][2] * top.z;
				vals[i] += const_conv_kernel3x3[1][0] * mid.x;
				vals[i] += const_conv_kernel3x3[1][1] * mid.y;
				vals[i] += const_conv_kernel3x3[1][2] * mid.z;
				vals[i] += const_conv_kernel3x3[2][0] * bot.x;
				vals[i] += const_conv_kernel3x3[2][1] * bot.y;
				vals[i] += const_conv_kernel3x3[2][2] * bot.z;
			}
		}
		reinterpret_cast<uchar2*>(&input[(tx * cols + ty)])[0] = make_uchar2(vals[0] >> 4, vals[1] >> 4);
	}
}

__global__ void k_1D_gf_3x3_shared(unsigned char* input, int rows, int cols)
{
	__shared__  unsigned char cache[34][34];

	int ty = blockIdx.x * blockDim.x + threadIdx.x;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	unsigned int cy = threadIdx.x + 1;
	unsigned int cx = threadIdx.y + 1;

	int new_val = 0;

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

	if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1)) {
		new_val += global_conv_kernel3x3[0][0] * cache[cx - 1][cy - 1];
		new_val += global_conv_kernel3x3[0][1] * cache[cx - 1][cy];
		new_val += global_conv_kernel3x3[0][2] * cache[cx - 1][cy + 1];

		new_val += global_conv_kernel3x3[1][0] * cache[cx][cy - 1];
		new_val += global_conv_kernel3x3[1][1] * cache[cx][cy];
		new_val += global_conv_kernel3x3[1][2] * cache[cx][cy + 1];

		new_val += global_conv_kernel3x3[2][0] * cache[cx + 1][cy - 1];
		new_val += global_conv_kernel3x3[2][1] * cache[cx + 1][cy];
		new_val += global_conv_kernel3x3[2][2] * cache[cx + 1][cy + 1];

		input[tx * cols + ty] = new_val >> 4;
	}
}

__global__ void k_1D_gf_3x3_load_balance16_shared(unsigned char* input, int rows, int cols)
{
	__shared__  unsigned char cache[34][514];

	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 16;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	int cy = threadIdx.x * 16 + 1;
	int cx = threadIdx.y + 1;

	cache[cx][cy] = input[tx * cols + ty];
	cache[cx][cy + 1] = input[tx * cols + ty + 1];
	cache[cx][cy + 2] = input[tx * cols + ty + 2];
	cache[cx][cy + 3] = input[tx * cols + ty + 3];
	cache[cx][cy + 4] = input[tx * cols + ty + 4];
	cache[cx][cy + 5] = input[tx * cols + ty + 5];
	cache[cx][cy + 6] = input[tx * cols + ty + 6];
	cache[cx][cy + 7] = input[tx * cols + ty + 7];
	cache[cx][cy + 8] = input[tx * cols + ty + 8];
	cache[cx][cy + 9] = input[tx * cols + ty + 9];
	cache[cx][cy + 10] = input[tx * cols + ty + 10];
	cache[cx][cy + 11] = input[tx * cols + ty + 11];
	cache[cx][cy + 12] = input[tx * cols + ty + 12];
	cache[cx][cy + 13] = input[tx * cols + ty + 13];
	cache[cx][cy + 14] = input[tx * cols + ty + 14];
	cache[cx][cy + 15] = input[tx * cols + ty + 15];

	if ((tx > 0 && tx < rows - 1) && (ty < cols - 1)) {
		if (cx == 1) { /*top row*/
			cache[0][cy] = input[((tx - 1) * cols + ty)];
			cache[0][cy + 1] = input[((tx - 1) * cols + ty + 1)];
			cache[0][cy + 2] = input[((tx - 1) * cols + ty + 2)];
			cache[0][cy + 3] = input[((tx - 1) * cols + ty + 3)];
			cache[0][cy + 4] = input[((tx - 1) * cols + ty + 4)];
			cache[0][cy + 5] = input[((tx - 1) * cols + ty + 5)];
			cache[0][cy + 6] = input[((tx - 1) * cols + ty + 6)];
			cache[0][cy + 7] = input[((tx - 1) * cols + ty + 7)];
			cache[0][cy + 8] = input[((tx - 1) * cols + ty + 8)];
			cache[0][cy + 9] = input[((tx - 1) * cols + ty + 9)];
			cache[0][cy + 10] = input[((tx - 1) * cols + ty + 10)];
			cache[0][cy + 11] = input[((tx - 1) * cols + ty + 11)];
			cache[0][cy + 12] = input[((tx - 1) * cols + ty + 12)];
			cache[0][cy + 13] = input[((tx - 1) * cols + ty + 13)];
			cache[0][cy + 14] = input[((tx - 1) * cols + ty + 14)];
			cache[0][cy + 15] = input[((tx - 1) * cols + ty + 15)];
		}
		if (cx == 32) { /*bottom row*/
			cache[33][cy] = input[((tx + 1) * cols + ty)];
			cache[33][cy + 1] = input[((tx + 1) * cols + ty + 1)];
			cache[33][cy + 2] = input[((tx + 1) * cols + ty + 2)];
			cache[33][cy + 3] = input[((tx + 1) * cols + ty + 3)];
			cache[33][cy + 4] = input[((tx + 1) * cols + ty + 4)];
			cache[33][cy + 5] = input[((tx + 1) * cols + ty + 5)];
			cache[33][cy + 6] = input[((tx + 1) * cols + ty + 6)];
			cache[33][cy + 7] = input[((tx + 1) * cols + ty + 7)];
			cache[33][cy + 8] = input[((tx + 1) * cols + ty + 8)];
			cache[33][cy + 9] = input[((tx + 1) * cols + ty + 9)];
			cache[33][cy + 10] = input[((tx + 1) * cols + ty + 10)];
			cache[33][cy + 11] = input[((tx + 1) * cols + ty + 11)];
			cache[33][cy + 12] = input[((tx + 1) * cols + ty + 12)];
			cache[33][cy + 13] = input[((tx + 1) * cols + ty + 13)];
			cache[33][cy + 14] = input[((tx + 1) * cols + ty + 14)];
			cache[33][cy + 15] = input[((tx + 1) * cols + ty + 15)];
		}
		if (cy == 1) {/*left column*/
			cache[cx][0] = input[((tx)*cols + ty - 1)];
		}
		if (cy == 497) {/*right column*/
			cache[cx][513] = input[((tx)*cols + ty + 16)];
		}

		__syncthreads();

		for (int i = 0; i < 16; i++) {
			int _ty = ty + i;
			int _cy = cy + i;
			int new_val = 0;
			if ( _ty < cols - 1) {
				new_val += global_conv_kernel3x3[0][0] * cache[cx - 1][_cy - 1];
				new_val += global_conv_kernel3x3[0][1] * cache[cx - 1][_cy];
				new_val += global_conv_kernel3x3[0][2] * cache[cx - 1][_cy + 1];

				new_val += global_conv_kernel3x3[1][0] * cache[cx][_cy - 1];
				new_val += global_conv_kernel3x3[1][1] * cache[cx][_cy];
				new_val += global_conv_kernel3x3[1][2] * cache[cx][_cy + 1];

				new_val += global_conv_kernel3x3[2][0] * cache[cx + 1][_cy - 1];
				new_val += global_conv_kernel3x3[2][1] * cache[cx + 1][_cy];
				new_val += global_conv_kernel3x3[2][2] * cache[cx + 1][_cy + 1];

				input[tx * cols + _ty] = new_val >> 4;
			}
		}
	}
}

__global__ void k_1D_gf_3x3_load_balance12_shared(unsigned char* input, int rows, int cols)
{
	__shared__  unsigned char cache[34][386];

	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 12;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	int cy = threadIdx.x * 12 + 1;
	int cx = threadIdx.y + 1;

	cache[cx][cy] = input[tx * cols + ty];
	cache[cx][cy + 1] = input[tx * cols + ty + 1];
	cache[cx][cy + 2] = input[tx * cols + ty + 2];
	cache[cx][cy + 3] = input[tx * cols + ty + 3];
	cache[cx][cy + 4] = input[tx * cols + ty + 4];
	cache[cx][cy + 5] = input[tx * cols + ty + 5];
	cache[cx][cy + 6] = input[tx * cols + ty + 6];
	cache[cx][cy + 7] = input[tx * cols + ty + 7];
	cache[cx][cy + 8] = input[tx * cols + ty + 8];
	cache[cx][cy + 9] = input[tx * cols + ty + 9];
	cache[cx][cy + 10] = input[tx * cols + ty + 10];
	cache[cx][cy + 11] = input[tx * cols + ty + 11];

	if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1)) {
		if (cx == 1) { /*top row*/
			cache[0][cy] = input[((tx - 1) * cols + ty)];
			cache[0][cy + 1] = input[((tx - 1) * cols + ty + 1)];
			cache[0][cy + 2] = input[((tx - 1) * cols + ty + 2)];
			cache[0][cy + 3] = input[((tx - 1) * cols + ty + 3)];
			cache[0][cy + 4] = input[((tx - 1) * cols + ty + 4)];
			cache[0][cy + 5] = input[((tx - 1) * cols + ty + 5)];
			cache[0][cy + 6] = input[((tx - 1) * cols + ty + 6)];
			cache[0][cy + 7] = input[((tx - 1) * cols + ty + 7)];
			cache[0][cy + 8] = input[((tx - 1) * cols + ty + 8)];
			cache[0][cy + 9] = input[((tx - 1) * cols + ty + 9)];
			cache[0][cy + 10] = input[((tx - 1) * cols + ty + 10)];
			cache[0][cy + 11] = input[((tx - 1) * cols + ty + 11)];
		}
		if (cx == 32) { /*bottom row*/
			cache[33][cy] = input[((tx + 1) * cols + ty)];
			cache[33][cy + 1] = input[((tx + 1) * cols + ty + 1)];
			cache[33][cy + 2] = input[((tx + 1) * cols + ty + 2)];
			cache[33][cy + 3] = input[((tx + 1) * cols + ty + 3)];
			cache[33][cy + 4] = input[((tx + 1) * cols + ty + 4)];
			cache[33][cy + 5] = input[((tx + 1) * cols + ty + 5)];
			cache[33][cy + 6] = input[((tx + 1) * cols + ty + 6)];
			cache[33][cy + 7] = input[((tx + 1) * cols + ty + 7)];
			cache[33][cy + 8] = input[((tx + 1) * cols + ty + 8)];
			cache[33][cy + 9] = input[((tx + 1) * cols + ty + 9)];
			cache[33][cy + 10] = input[((tx + 1) * cols + ty + 10)];
			cache[33][cy + 11] = input[((tx + 1) * cols + ty + 11)];
		}
		if (cy == 1) {/*left column*/
			cache[cx][0] = input[((tx)*cols + ty - 1)];
		}
		if (cy == 373) {/*right column*/
			cache[cx][385] = input[((tx)*cols + ty + 12)];
		}

		__syncthreads();

		for (int i = 0; i < 12; i++) {
			int _ty = ty + i;
			int _cy = cy + i;
			int new_val = 0;
			if (_ty < cols - 1) {
				new_val += global_conv_kernel3x3[0][0] * cache[cx - 1][_cy - 1];
				new_val += global_conv_kernel3x3[0][1] * cache[cx - 1][_cy];
				new_val += global_conv_kernel3x3[0][2] * cache[cx - 1][_cy + 1];

				new_val += global_conv_kernel3x3[1][0] * cache[cx][_cy - 1];
				new_val += global_conv_kernel3x3[1][1] * cache[cx][_cy];
				new_val += global_conv_kernel3x3[1][2] * cache[cx][_cy + 1];

				new_val += global_conv_kernel3x3[2][0] * cache[cx + 1][_cy - 1];
				new_val += global_conv_kernel3x3[2][1] * cache[cx + 1][_cy];
				new_val += global_conv_kernel3x3[2][2] * cache[cx + 1][_cy + 1];

				input[tx * cols + _ty] = new_val >> 4;
			}
		}
	}
}

__global__ void k_1D_gf_3x3_load_balance8_shared(unsigned char* input, int rows, int cols)
{
	__shared__  unsigned char cache[34][260];

	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	int cy = threadIdx.x * 8 + 1;
	int cx = threadIdx.y + 1;

	cache[cx][cy] = input[tx * cols + ty];
	cache[cx][cy + 1] = input[tx * cols + ty + 1];
	cache[cx][cy + 2] = input[tx * cols + ty + 2];
	cache[cx][cy + 3] = input[tx * cols + ty + 3];
	cache[cx][cy + 4] = input[tx * cols + ty + 4];
	cache[cx][cy + 5] = input[tx * cols + ty + 5];
	cache[cx][cy + 6] = input[tx * cols + ty + 6];
	cache[cx][cy + 7] = input[tx * cols + ty + 7];

	if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1)) {
		if (cx == 1) { /*top row*/
			cache[0][cy] = input[((tx - 1) * cols + ty)];
			cache[0][cy + 1] = input[((tx - 1) * cols + ty + 1)];
			cache[0][cy + 2] = input[((tx - 1) * cols + ty + 2)];
			cache[0][cy + 3] = input[((tx - 1) * cols + ty + 3)];
			cache[0][cy + 4] = input[((tx - 1) * cols + ty + 4)];
			cache[0][cy + 5] = input[((tx - 1) * cols + ty + 5)];
			cache[0][cy + 6] = input[((tx - 1) * cols + ty + 6)];
			cache[0][cy + 7] = input[((tx - 1) * cols + ty + 7)];
		}
		if (cx == 32) { /*bottom row*/
			cache[33][cy] = input[((tx + 1) * cols + ty)];
			cache[33][cy + 1] = input[((tx + 1) * cols + ty + 1)];
			cache[33][cy + 2] = input[((tx + 1) * cols + ty + 2)];
			cache[33][cy + 3] = input[((tx + 1) * cols + ty + 3)];
			cache[33][cy + 4] = input[((tx + 1) * cols + ty + 4)];
			cache[33][cy + 5] = input[((tx + 1) * cols + ty + 5)];
			cache[33][cy + 6] = input[((tx + 1) * cols + ty + 6)];
			cache[33][cy + 7] = input[((tx + 1) * cols + ty + 7)];
		}
		if (cy == 1) {/*left column*/
			cache[cx][0] = input[((tx)*cols + ty - 1)];
		}
		if (cy == 249) {/*right column*/
			cache[cx][257] = input[((tx)*cols + ty + 8)];
		}
		__syncthreads();

		for (int i = 0; i < 8; i++) {
			int _ty = ty + i;
			int _cy = cy + i;
			int new_val = 0;
			if (_ty < cols - 1) {
				new_val += global_conv_kernel3x3[0][0] * cache[cx - 1][_cy - 1];
				new_val += global_conv_kernel3x3[0][1] * cache[cx - 1][_cy];
				new_val += global_conv_kernel3x3[0][2] * cache[cx - 1][_cy + 1];

				new_val += global_conv_kernel3x3[1][0] * cache[cx][_cy - 1];
				new_val += global_conv_kernel3x3[1][1] * cache[cx][_cy];
				new_val += global_conv_kernel3x3[1][2] * cache[cx][_cy + 1];

				new_val += global_conv_kernel3x3[2][0] * cache[cx + 1][_cy - 1];
				new_val += global_conv_kernel3x3[2][1] * cache[cx + 1][_cy];
				new_val += global_conv_kernel3x3[2][2] * cache[cx + 1][_cy + 1];

				input[tx * cols + _ty] = new_val >> 4;
			}
		}
	}
}

__global__ void k_1D_gf_3x3_load_balance4_shared(unsigned char* input, int rows, int cols)
{
	__shared__  unsigned char cache[34][130];

	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	int cy = threadIdx.x * 4 + 1;
	int cx = threadIdx.y + 1;

	cache[cx][cy] = input[tx * cols + ty];
	cache[cx][cy + 1] = input[tx * cols + ty + 1];
	cache[cx][cy + 2] = input[tx * cols + ty + 2];
	cache[cx][cy + 3] = input[tx * cols + ty + 3];

	if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1)) {
		if (cx == 1) { /*top row*/
			cache[0][cy] = input[((tx - 1) * cols + ty)];
			cache[0][cy + 1] = input[((tx - 1) * cols + ty + 1)];
			cache[0][cy + 2] = input[((tx - 1) * cols + ty + 2)];
			cache[0][cy + 3] = input[((tx - 1) * cols + ty + 3)];
		}
		if (cx == 32) { /*bottom row*/
			cache[33][cy] = input[((tx + 1) * cols + ty)];
			cache[33][cy + 1] = input[((tx + 1) * cols + ty + 1)];
			cache[33][cy + 2] = input[((tx + 1) * cols + ty + 2)];
			cache[33][cy + 3] = input[((tx + 1) * cols + ty + 3)];
		}
		if (cy == 1) {/*left column*/
			cache[cx][0] = input[((tx)*cols + ty - 1)];
		}
		if (cy == 125) {/*right column*/
			cache[cx][129] = input[((tx)*cols + ty + 4)];
		}
		__syncthreads();

		for (int i = 0; i < 4; i++) {
			int _ty = ty + i;
			int _cy = cy + i;
			int new_val = 0;
			if (_ty < cols - 1) {
				new_val += global_conv_kernel3x3[0][0] * cache[cx - 1][_cy - 1];
				new_val += global_conv_kernel3x3[0][1] * cache[cx - 1][_cy];
				new_val += global_conv_kernel3x3[0][2] * cache[cx - 1][_cy + 1];

				new_val += global_conv_kernel3x3[1][0] * cache[cx][_cy - 1];
				new_val += global_conv_kernel3x3[1][1] * cache[cx][_cy];
				new_val += global_conv_kernel3x3[1][2] * cache[cx][_cy + 1];

				new_val += global_conv_kernel3x3[2][0] * cache[cx + 1][_cy - 1];
				new_val += global_conv_kernel3x3[2][1] * cache[cx + 1][_cy];
				new_val += global_conv_kernel3x3[2][2] * cache[cx + 1][_cy + 1];

				input[tx * cols + _ty] = new_val >> 4;
			}
		}
	}
}

__global__ void k_1D_gf_3x3_load_balance2_shared(unsigned char* input, int rows, int cols)
{
	__shared__  unsigned char cache[34][66];

	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	int cy = threadIdx.x * 2 + 1;
	int cx = threadIdx.y + 1;

	cache[cx][cy] = input[tx * cols + ty];
	cache[cx][cy + 1] = input[tx * cols + ty + 1];

	if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1)) {
		if (cx == 1) { /*top row*/
			cache[0][cy] = input[((tx - 1) * cols + ty)];
			cache[0][cy + 1] = input[((tx - 1) * cols + ty + 1)];
		}
		if (cx == 32) { /*bottom row*/
			cache[33][cy] = input[((tx + 1) * cols + ty)];
			cache[33][cy + 1] = input[((tx + 1) * cols + ty + 1)];
		}
		if (cy == 1) {/*left column*/
			cache[cx][0] = input[((tx)*cols + ty - 1)];
		}
		if (cy == 63) {/*right column*/
			cache[cx][65] = input[((tx)*cols + ty + 2)];
		}
		__syncthreads();

		for (int i = 0; i < 2; i++) {
			int _ty = ty + i;
			int _cy = cy + i;
			int new_val = 0;
			if ( _ty < cols - 1) {
				new_val += global_conv_kernel3x3[0][0] * cache[cx - 1][_cy - 1];
				new_val += global_conv_kernel3x3[0][1] * cache[cx - 1][_cy];
				new_val += global_conv_kernel3x3[0][2] * cache[cx - 1][_cy + 1];

				new_val += global_conv_kernel3x3[1][0] * cache[cx][_cy - 1];
				new_val += global_conv_kernel3x3[1][1] * cache[cx][_cy];
				new_val += global_conv_kernel3x3[1][2] * cache[cx][_cy + 1];

				new_val += global_conv_kernel3x3[2][0] * cache[cx + 1][_cy - 1];
				new_val += global_conv_kernel3x3[2][1] * cache[cx + 1][_cy];
				new_val += global_conv_kernel3x3[2][2] * cache[cx + 1][_cy + 1];

				input[tx * cols + _ty] = new_val >> 4;
			}
		}
	}
}

__global__ void k_1D_gf_3x3_vectorized16_shared(unsigned char* input, int rows, int cols)
{
	__shared__  unsigned char cache[34][514];

	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 16;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	int cy = threadIdx.x * 16 + 1;
	int cx = threadIdx.y + 1;

	uchar4 u4;
	u4 = reinterpret_cast<uchar4*>(&input[tx * cols + ty])[0];

	cache[cx][cy] = u4.x;
	cache[cx][cy + 1] = u4.y;
	cache[cx][cy + 2] = u4.z;
	cache[cx][cy + 3] = u4.w;

	u4 = reinterpret_cast<uchar4*>(&input[tx * cols + ty + 4])[0];
	cache[cx][cy + 4] = u4.x;
	cache[cx][cy + 5] = u4.y;
	cache[cx][cy + 6] = u4.z;
	cache[cx][cy + 7] = u4.w;

	u4 = reinterpret_cast<uchar4*>(&input[tx * cols + ty + 8])[0];
	cache[cx][cy + 8] = u4.x;
	cache[cx][cy + 9] = u4.y;
	cache[cx][cy + 10] = u4.z;
	cache[cx][cy + 11] = u4.w;

	u4 = reinterpret_cast<uchar4*>(&input[tx * cols + ty + 12])[0];
	cache[cx][cy + 12] = u4.x;
	cache[cx][cy + 13] = u4.y;
	cache[cx][cy + 14] = u4.z;
	cache[cx][cy + 15] = u4.w;

	if ((tx > 0 && tx < rows - 1) && (ty < cols - 1)) {
		if (cx == 1) { /*top row*/
			u4 = reinterpret_cast<uchar4*>(&input[(tx - 1) * cols + ty])[0];
			cache[0][cy] = u4.x;
			cache[0][cy + 1] = u4.y;
			cache[0][cy + 2] = u4.z;
			cache[0][cy + 3] = u4.w;

			u4 = reinterpret_cast<uchar4*>(&input[(tx - 1) * cols + ty + 4])[0];
			cache[0][cy + 4] = u4.x;
			cache[0][cy + 5] = u4.y;
			cache[0][cy + 6] = u4.z;
			cache[0][cy + 7] = u4.w;

			u4 = reinterpret_cast<uchar4*>(&input[(tx - 1) * cols + ty + 8])[0];
			cache[0][cy + 8] = u4.x;
			cache[0][cy + 9] = u4.y;
			cache[0][cy + 10] = u4.z;
			cache[0][cy + 11] = u4.w;

			u4 = reinterpret_cast<uchar4*>(&input[(tx - 1) * cols + ty + 12])[0];
			cache[0][cy + 12] = u4.x;
			cache[0][cy + 13] = u4.y;
			cache[0][cy + 14] = u4.z;
			cache[0][cy + 15] = u4.w;
		}
		if (cx == 32) { /*bottom row*/
			u4 = reinterpret_cast<uchar4*>(&input[(tx + 1) * cols + ty])[0];
			cache[33][cy] = u4.x;
			cache[33][cy + 1] = u4.y;
			cache[33][cy + 2] = u4.z;
			cache[33][cy + 3] = u4.w;
			u4 = reinterpret_cast<uchar4*>(&input[(tx + 1) * cols + ty + 4])[0];
			cache[33][cy + 4] = u4.x;
			cache[33][cy + 5] = u4.y;
			cache[33][cy + 6] = u4.z;
			cache[33][cy + 7] = u4.w;
			u4 = reinterpret_cast<uchar4*>(&input[(tx + 1) * cols + ty + 8])[0];
			cache[33][cy + 8] = u4.x;
			cache[33][cy + 9] = u4.y;
			cache[33][cy + 10] = u4.z;
			cache[33][cy + 11] = u4.w;
			u4 = reinterpret_cast<uchar4*>(&input[(tx + 1) * cols + ty + 12])[0];
			cache[33][cy + 12] = u4.x;
			cache[33][cy + 13] = u4.y;
			cache[33][cy + 14] = u4.z;
			cache[33][cy + 15] = u4.w;
		}
		if (cy == 1) {/*left column*/
			cache[cx][0] = input[((tx)*cols + ty - 1)];
		}
		if (cy == 497) {/*right column*/
			cache[cx][513] = input[((tx)*cols + ty + 16)];
		}

		__syncthreads();
		int vals[16] = { 0 };
		for (int i = 0; i < 16; i++) {
			int _ty = ty + i;
			int _cy = cy + i;
			if (_ty < cols - 1) {
				vals[i] += global_conv_kernel3x3[0][0] * cache[cx - 1][_cy - 1];
				vals[i] += global_conv_kernel3x3[0][1] * cache[cx - 1][_cy];
				vals[i] += global_conv_kernel3x3[0][2] * cache[cx - 1][_cy + 1];
				vals[i] += global_conv_kernel3x3[1][0] * cache[cx][_cy - 1];
				vals[i] += global_conv_kernel3x3[1][1] * cache[cx][_cy];
				vals[i] += global_conv_kernel3x3[1][2] * cache[cx][_cy + 1];
				vals[i] += global_conv_kernel3x3[2][0] * cache[cx + 1][_cy - 1];
				vals[i] += global_conv_kernel3x3[2][1] * cache[cx + 1][_cy];
				vals[i] += global_conv_kernel3x3[2][2] * cache[cx + 1][_cy + 1];
			}
		}
		reinterpret_cast<uchar4*>(&input[(tx * cols + ty)])[0] = make_uchar4(vals[0] >> 4, vals[1] >> 4, vals[2] >> 4, vals[3] >> 4);
		reinterpret_cast<uchar4*>(&input[(tx * cols + ty + 4)])[0] = make_uchar4(vals[4] >> 4, vals[5] >> 4, vals[6] >> 4, vals[7] >> 4);
		reinterpret_cast<uchar4*>(&input[(tx * cols + ty + 8)])[0] = make_uchar4(vals[8] >> 4, vals[9] >> 4, vals[10] >> 4, vals[11] >> 4);
		reinterpret_cast<uchar4*>(&input[(tx * cols + ty + 12)])[0] = make_uchar4(vals[12] >> 4, vals[13] >> 4, vals[14] >> 4, vals[15] >> 4);
	}
}

__global__ void k_1D_gf_3x3_vectorized12_shared(unsigned char* input, int rows, int cols)
{
	__shared__  unsigned char cache[34][386];

	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 12;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	int cy = threadIdx.x * 12 + 1;
	int cx = threadIdx.y + 1;
	
	uchar4 u4;
	u4 = reinterpret_cast<uchar4*>(&input[tx * cols + ty])[0];

	cache[cx][cy] = u4.x;
	cache[cx][cy + 1] = u4.y;
	cache[cx][cy + 2] = u4.z;
	cache[cx][cy + 3] = u4.w;

	u4 = reinterpret_cast<uchar4*>(&input[tx * cols + ty + 4])[0];
	cache[cx][cy + 4] = u4.x;
	cache[cx][cy + 5] = u4.y;
	cache[cx][cy + 6] = u4.z;
	cache[cx][cy + 7] = u4.w;

	u4 = reinterpret_cast<uchar4*>(&input[tx * cols + ty + 8])[0];
	cache[cx][cy + 8] = u4.x;
	cache[cx][cy + 9] = u4.y;
	cache[cx][cy + 10] = u4.z;
	cache[cx][cy + 11] = u4.w;

	if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1)) {
		if (cx == 1) { /*top row*/
			u4 = reinterpret_cast<uchar4*>(&input[(tx - 1 ) * cols + ty])[0];
			cache[0][cy] = u4.x;
			cache[0][cy + 1] = u4.y;
			cache[0][cy + 2] = u4.z;
			cache[0][cy + 3] = u4.w;

			u4 = reinterpret_cast<uchar4*>(&input[(tx - 1) * cols + ty + 4])[0];
			cache[0][cy + 4] = u4.x;
			cache[0][cy + 5] = u4.y;
			cache[0][cy + 6] = u4.z;
			cache[0][cy + 7] = u4.w;

			u4 = reinterpret_cast<uchar4*>(&input[(tx - 1) * cols + ty + 8])[0];
			cache[0][cy + 8] = u4.x;
			cache[0][cy + 9] = u4.y;
			cache[0][cy + 10] = u4.z; 
			cache[0][cy + 11] = u4.w; 
		}
		if (cx == 32) { /*bottom row*/
			u4 = reinterpret_cast<uchar4*>(&input[(tx + 1) * cols + ty])[0];
			cache[33][cy] = u4.x;
			cache[33][cy + 1] = u4.y;
			cache[33][cy + 2] = u4.z;
			cache[33][cy + 3] = u4.w;
			u4 = reinterpret_cast<uchar4*>(&input[(tx + 1) * cols + ty + 4])[0];
			cache[33][cy + 4] = u4.x;
			cache[33][cy + 5] = u4.y;
			cache[33][cy + 6] = u4.z;
			cache[33][cy + 7] = u4.w;
			u4 = reinterpret_cast<uchar4*>(&input[(tx + 1) * cols + ty + 8])[0];
			cache[33][cy + 8] = u4.x;
			cache[33][cy + 9] = u4.y;
			cache[33][cy + 10] = u4.z;
			cache[33][cy + 11] = u4.w;
		}
		if (cy == 1) {/*left column*/
			cache[cx][0] = input[((tx)*cols + ty - 1)];
		}
		if (cy == 373) {/*right column*/
			cache[cx][385] = input[((tx)*cols + ty + 12)];
		}
		__syncthreads();
		int vals[12] = { 0 };
		for (int i = 0; i < 12; i++) {
			int _ty = ty + i;
			int _cy = cy + i;
			if (_ty < cols - 1) {
				vals[i] += global_conv_kernel3x3[0][0] * cache[cx - 1][_cy - 1];
				vals[i] += global_conv_kernel3x3[0][1] * cache[cx - 1][_cy];
				vals[i] += global_conv_kernel3x3[0][2] * cache[cx - 1][_cy + 1];
				vals[i] += global_conv_kernel3x3[1][0] * cache[cx][_cy - 1];
				vals[i] += global_conv_kernel3x3[1][1] * cache[cx][_cy];
				vals[i] += global_conv_kernel3x3[1][2] * cache[cx][_cy + 1];
				vals[i] += global_conv_kernel3x3[2][0] * cache[cx + 1][_cy - 1];
				vals[i] += global_conv_kernel3x3[2][1] * cache[cx + 1][_cy];
				vals[i] += global_conv_kernel3x3[2][2] * cache[cx + 1][_cy + 1];
			}
		}
		reinterpret_cast<uchar4*>(&input[(tx * cols + ty)])[0] = make_uchar4(vals[0] >> 4, vals[1] >> 4, vals[2] >> 4, vals[3] >> 4);
		reinterpret_cast<uchar4*>(&input[(tx * cols + ty + 4)])[0] = make_uchar4(vals[4] >> 4, vals[5] >> 4, vals[6] >> 4, vals[7] >> 4);
		reinterpret_cast<uchar4*>(&input[(tx * cols + ty + 8)])[0] = make_uchar4(vals[8] >> 4, vals[9] >> 4, vals[10] >> 4, vals[11] >> 4);
	}
}

__global__ void k_1D_gf_3x3_vectorized8_shared(unsigned char* input, int rows, int cols)
{
	__shared__  unsigned char cache[34][260];

	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	int cy = threadIdx.x * 8 + 1;
	int cx = threadIdx.y + 1;

	uchar4 u4;

	u4 = reinterpret_cast<uchar4*>(&input[tx * cols + ty])[0];

	cache[cx][cy] = u4.x;
	cache[cx][cy + 1] = u4.y;
	cache[cx][cy + 2] = u4.z;
	cache[cx][cy + 3] = u4.w;

	u4 = reinterpret_cast<uchar4*>(&input[tx * cols + ty + 4])[0];
	cache[cx][cy + 4] = u4.x;
	cache[cx][cy + 5] = u4.y;
	cache[cx][cy + 6] = u4.z;
	cache[cx][cy + 7] = u4.w;

	if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1)) {
		if (cx == 1) { /*top row*/
			u4 = reinterpret_cast<uchar4*>(&input[(tx - 1) * cols + ty])[0];
			cache[0][cy] = u4.x;
			cache[0][cy + 1] = u4.y;
			cache[0][cy + 2] = u4.z;
			cache[0][cy + 3] = u4.w;
			u4 = reinterpret_cast<uchar4*>(&input[(tx - 1) * cols + ty + 4])[0];
			cache[0][cy + 4] = u4.x;
			cache[0][cy + 5] = u4.y;
			cache[0][cy + 6] = u4.z;
			cache[0][cy + 7] = u4.w;
		}
		if (cx == 32) { /*bottom row*/
			u4 = reinterpret_cast<uchar4*>(&input[(tx + 1) * cols + ty])[0];
			cache[33][cy] = u4.x;
			cache[33][cy + 1] = u4.y;
			cache[33][cy + 2] = u4.z;
			cache[33][cy + 3] = u4.w;
			u4 = reinterpret_cast<uchar4*>(&input[(tx + 1) * cols + ty + 4])[0];
			cache[33][cy + 4] = u4.x;
			cache[33][cy + 5] = u4.y;
			cache[33][cy + 6] = u4.z;
			cache[33][cy + 7] = u4.w;
		}
		if (cy == 1) {/*left column*/
			cache[cx][0] = input[((tx)*cols + ty - 1)];
		}
		if (cy == 249) {/*right column*/
			cache[cx][257] = input[((tx)*cols + ty + 8)];
		}
		__syncthreads();
		int vals[8] = { 0 };
		for (int i = 0; i < 8; i++) {
			int _ty = ty + i;
			int _cy = cy + i;
			if (_ty < cols - 1) {
				vals[i] += global_conv_kernel3x3[0][0] * cache[cx - 1][_cy - 1];
				vals[i] += global_conv_kernel3x3[0][1] * cache[cx - 1][_cy];
				vals[i] += global_conv_kernel3x3[0][2] * cache[cx - 1][_cy + 1];
				vals[i] += global_conv_kernel3x3[1][0] * cache[cx][_cy - 1];
				vals[i] += global_conv_kernel3x3[1][1] * cache[cx][_cy];
				vals[i] += global_conv_kernel3x3[1][2] * cache[cx][_cy + 1];
				vals[i] += global_conv_kernel3x3[2][0] * cache[cx + 1][_cy - 1];
				vals[i] += global_conv_kernel3x3[2][1] * cache[cx + 1][_cy];
				vals[i] += global_conv_kernel3x3[2][2] * cache[cx + 1][_cy + 1];
			}
		}
		reinterpret_cast<uchar4*>(&input[(tx * cols + ty)])[0] = make_uchar4(vals[0] >> 4, vals[1] >> 4, vals[2] >> 4, vals[3] >> 4);
		reinterpret_cast<uchar4*>(&input[(tx * cols + ty + 4)])[0] = make_uchar4(vals[4] >> 4, vals[5] >> 4, vals[6] >> 4, vals[7] >> 4);
	}
}

__global__ void k_1D_gf_3x3_vectorized4_shared(unsigned char* input, int rows, int cols)
{
	__shared__  unsigned char cache[34][130];

	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	int cy = threadIdx.x * 4 + 1;
	int cx = threadIdx.y + 1;

	uchar4 u4;

	u4 = reinterpret_cast<uchar4*>(&input[tx * cols + ty])[0];
	cache[cx][cy] = u4.x;
	cache[cx][cy + 1] = u4.y;
	cache[cx][cy + 2] = u4.z;
	cache[cx][cy + 3] = u4.w;

	if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1)) {
		if (cx == 1) { /*top row*/
			u4 = reinterpret_cast<uchar4*>(&input[(tx - 1) * cols + ty])[0];
			cache[0][cy] = u4.x;
			cache[0][cy + 1] = u4.y;
			cache[0][cy + 2] = u4.z;
			cache[0][cy + 3] = u4.w;
		}
		if (cx == 32) { /*bottom row*/
			u4 = reinterpret_cast<uchar4*>(&input[(tx + 1) * cols + ty])[0];
			cache[33][cy] = u4.x;
			cache[33][cy + 1] = u4.y;
			cache[33][cy + 2] = u4.z;
			cache[33][cy + 3] = u4.w;
		}
		if (cy == 1) {/*left column*/
			cache[cx][0] = input[(tx * cols + ty - 1)];
		}
		if (cy == 125) {/*right column*/
			cache[cx][129] = input[(tx * cols + ty + 4)];
		}
		__syncthreads();
		int vals[4] = { 0 };
		for (int i = 0; i < 4; i++) {
			int _ty = ty + i;
			int _cy = cy + i;
			if (_ty < cols - 1) {
				uchar3 top = reinterpret_cast<uchar3*>(&cache[cx - 1][_cy - 1])[0];
				uchar3 mid = reinterpret_cast<uchar3*>(&cache[cx][_cy - 1])[0];
				uchar3 bot = reinterpret_cast<uchar3*>(&cache[cx + 1][_cy - 1])[0];

				vals[i] += global_conv_kernel3x3[0][0] * top.x;
				vals[i] += global_conv_kernel3x3[0][1] * top.y;
				vals[i] += global_conv_kernel3x3[0][2] * top.z;
				vals[i] += global_conv_kernel3x3[1][0] * mid.x;
				vals[i] += global_conv_kernel3x3[1][1] * mid.y;
				vals[i] += global_conv_kernel3x3[1][2] * mid.z;
				vals[i] += global_conv_kernel3x3[2][0] * bot.x;
				vals[i] += global_conv_kernel3x3[2][1] * bot.y;
				vals[i] += global_conv_kernel3x3[2][2] * bot.z;

			}
		}
		reinterpret_cast<uchar4*>(&input[(tx * cols + ty)])[0] = make_uchar4(vals[0] >> 4, vals[1] >> 4, vals[2] >> 4, vals[3] >> 4);
	}
}


__global__ void k_1D_gf_3x3_vectorized2_shared(unsigned char* input, int rows, int cols)
{
	__shared__  unsigned char cache[34][66];

	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	int cy = threadIdx.x * 2 + 1;
	int cx = threadIdx.y + 1;

	uchar2 u2;
	u2 = reinterpret_cast<uchar2*>(&input[tx * cols + ty])[0];

	cache[cx][cy] = u2.x;
	cache[cx][cy + 1] = u2.y;

	if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1)) {
		if (cx == 1) { /*top row*/
			u2 = reinterpret_cast<uchar2*>(&input[(tx - 1) * cols + ty])[0];
			cache[0][cy] = u2.x;
			cache[0][cy + 1] = u2.y;
		}
		if (cx == 32) { /*bottom row*/
			u2 = reinterpret_cast<uchar2*>(&input[(tx + 1) * cols + ty])[0];
			cache[33][cy] = u2.x;
			cache[33][cy + 1] = u2.y;
		}
		if (cy == 1) {/*left column*/
			cache[cx][0] = input[(tx * cols + ty - 1)];
		}
		if (cy == 63) {/*right column*/
			cache[cx][65] = input[((tx)*cols + ty + 2)];
		}
		__syncthreads();
		int vals[2] = { 0 };

		for (int i = 0; i < 2; i++) {
			int _ty = ty + i;
			int _cy = cy + i;
			int new_val = 0;

			if (_ty < cols - 1) {
				uchar3 top = reinterpret_cast<uchar3*>(&cache[cx - 1][_cy - 1])[0];
				uchar3 mid = reinterpret_cast<uchar3*>(&cache[cx][_cy - 1])[0];
				uchar3 bot = reinterpret_cast<uchar3*>(&cache[cx + 1][_cy - 1])[0];

				vals[i] += global_conv_kernel3x3[0][0] * top.x;
				vals[i] += global_conv_kernel3x3[0][1] * top.y;
				vals[i] += global_conv_kernel3x3[0][2] * top.z;
				vals[i] += global_conv_kernel3x3[1][0] * mid.x;
				vals[i] += global_conv_kernel3x3[1][1] * mid.y;
				vals[i] += global_conv_kernel3x3[1][2] * mid.z;
				vals[i] += global_conv_kernel3x3[2][0] * bot.x;
				vals[i] += global_conv_kernel3x3[2][1] * bot.y;
				vals[i] += global_conv_kernel3x3[2][2] * bot.z;

				input[tx * cols + _ty] = new_val >> 4;
			}
		}
		reinterpret_cast<uchar2*>(&input[(tx * cols + ty)])[0] = make_uchar2(vals[0] >> 4, vals[1] >> 4);
	}
}

float gf_1d_gpu(cv::Mat* output_img, GAUSSIAN ver)
{
	unsigned char* gpu_input = nullptr;
	unsigned char* output = output_img->data;

	int cols = (*output_img).cols;
	int rows = (*output_img).rows;
	int size = cols * rows * sizeof(unsigned char);

	unsigned char conv_kernel3x3[3][3] = { {1, 2, 1}, {2, 4, 2}, {1, 2, 1} };

	dim3 block(16,16);
	dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
	dim3 grid2(((cols / 2) + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
	dim3 grid4(((cols / 4) + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
	dim3 grid8(((cols / 8) + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
	dim3 grid12(((cols / 12) + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
	dim3 grid16(((cols / 16) + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
	dim3 grid32(((cols / 32) + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	cudaHostRegister(output, size, cudaHostRegisterPortable);
	CHECK_CUDA_ERROR(cudaMalloc((unsigned char**)&gpu_input, size));
	CHECK_CUDA_ERROR(cudaMemcpy(gpu_input, output, size, cudaMemcpyHostToDevice));

	switch (ver)
	{
	default:
		break;
	case GAUSSIAN_3x3_global:
		k_1D_gf_3x3_global << <grid, block >> > (gpu_input, rows, cols);
		break;
	case GAUSSIAN_3x3_local:
		k_1D_gf_3x3_local << <grid, block >> > (gpu_input, rows, cols);
		break;
	case GAUSSIAN_3x3_constant:
		CHECK_CUDA_ERROR(cudaMemcpyToSymbol(const_conv_kernel3x3, conv_kernel3x3, sizeof(unsigned char) * 3 * 3));
		k_1D_gf_3x3_constant << <grid, block >> > (gpu_input, rows, cols);
		break;
	case GAUSSIAN_3x3_shared:
		k_1D_gf_3x3_shared << <grid, block >> > (gpu_input, rows, cols);
		break;
	case GAUSSIAN_3x3_load_balance32_global:
		k_1D_gf_3x3_load_balance32_global << <grid32, block >> > (gpu_input, rows, cols);
		break;
	case GAUSSIAN_3x3_load_balance16_global:
		k_1D_gf_3x3_load_balance16_global << <grid16, block >> > (gpu_input, rows, cols);
		break;
	case GAUSSIAN_3x3_load_balance12_global:
		k_1D_gf_3x3_load_balance12_global << <grid12, block >> > (gpu_input, rows, cols);
		break;
	case GAUSSIAN_3x3_load_balance8_global:
		k_1D_gf_3x3_load_balance8_global << <grid8, block >> > (gpu_input, rows, cols);
		break;
	case GAUSSIAN_3x3_load_balance4_global:
		k_1D_gf_3x3_load_balance4_global << <grid4, block >> > (gpu_input, rows, cols);
		break;
	case GAUSSIAN_3x3_load_balance2_global:
		k_1D_gf_3x3_load_balance2_global << <grid2, block >> > (gpu_input, rows, cols);
		break;
	case GAUSSIAN_3x3_vectorized32_global:
		k_1D_gf_3x3_vectorized32_global << <grid32, block >> > (gpu_input, rows, cols);
		break;
	case GAUSSIAN_3x3_vectorized16_global:
		k_1D_gf_3x3_vectorized16_global << <grid16, block >> > (gpu_input, rows, cols);
		break;
	case GAUSSIAN_3x3_vectorized12_global:
		k_1D_gf_3x3_vectorized12_global << <grid12, block >> > (gpu_input, rows, cols);
		break;
	case GAUSSIAN_3x3_vectorized8_global:
		k_1D_gf_3x3_vectorized8_global << <grid8, block >> > (gpu_input, rows, cols);
		break;
	case GAUSSIAN_3x3_vectorized4_global:
		k_1D_gf_3x3_vectorized4_global << <grid4, block >> > (gpu_input, rows, cols);
		break;
	case GAUSSIAN_3x3_vectorized2_global:
		k_1D_gf_3x3_vectorized2_global << <grid2, block >> > (gpu_input, rows, cols);
		break;
	case GAUSSIAN_3x3_load_balance16_local:
		k_1D_gf_3x3_load_balance16_local << <grid16, block >> > (gpu_input, rows, cols);
		break;
	case GAUSSIAN_3x3_load_balance12_local:
		k_1D_gf_3x3_load_balance12_local << <grid12, block >> > (gpu_input, rows, cols);
		break;
	case GAUSSIAN_3x3_load_balance8_local:
		k_1D_gf_3x3_load_balance8_local << <grid8, block >> > (gpu_input, rows, cols);
		break;
	case GAUSSIAN_3x3_load_balance4_local:
		k_1D_gf_3x3_load_balance4_local << <grid4, block >> > (gpu_input, rows, cols);
		break;
	case GAUSSIAN_3x3_load_balance2_local:
		k_1D_gf_3x3_load_balance2_local << <grid2, block >> > (gpu_input, rows, cols);
		break;
	case GAUSSIAN_3x3_vectorized16_local:
		k_1D_gf_3x3_vectorized16_local << <grid16, block >> > (gpu_input, rows, cols);
		break;
	case GAUSSIAN_3x3_vectorized12_local:
		k_1D_gf_3x3_vectorized12_local << <grid12, block >> > (gpu_input, rows, cols);
		break;
	case GAUSSIAN_3x3_vectorized8_local:
		k_1D_gf_3x3_vectorized8_local << <grid8, block >> > (gpu_input, rows, cols);
		break;
	case GAUSSIAN_3x3_vectorized4_local:
		k_1D_gf_3x3_vectorized4_local << <grid4, block >> > (gpu_input, rows, cols);
		break;
	case GAUSSIAN_3x3_vectorized2_local:
		k_1D_gf_3x3_vectorized2_local << <grid2, block >> > (gpu_input, rows, cols);
		break;
	case GAUSSIAN_3x3_load_balance16_constant:
		CHECK_CUDA_ERROR(cudaMemcpyToSymbol(const_conv_kernel3x3, conv_kernel3x3, sizeof(unsigned char) * 3 * 3));
		k_1D_gf_3x3_load_balance16_constant << <grid16, block >> > (gpu_input, rows, cols);
		break;
	case GAUSSIAN_3x3_load_balance12_constant:
		CHECK_CUDA_ERROR(cudaMemcpyToSymbol(const_conv_kernel3x3, conv_kernel3x3, sizeof(unsigned char) * 3 * 3));
		k_1D_gf_3x3_load_balance12_constant << <grid12, block >> > (gpu_input, rows, cols);
		break;
	case GAUSSIAN_3x3_load_balance8_constant:
		CHECK_CUDA_ERROR(cudaMemcpyToSymbol(const_conv_kernel3x3, conv_kernel3x3, sizeof(unsigned char) * 3 * 3));
		k_1D_gf_3x3_load_balance8_constant << <grid8, block >> > (gpu_input, rows, cols);
		break;
	case GAUSSIAN_3x3_load_balance4_constant:
		CHECK_CUDA_ERROR(cudaMemcpyToSymbol(const_conv_kernel3x3, conv_kernel3x3, sizeof(unsigned char) * 3 * 3));
		k_1D_gf_3x3_load_balance4_constant << <grid4, block >> > (gpu_input, rows, cols);
		break;
	case GAUSSIAN_3x3_load_balance2_constant:
		CHECK_CUDA_ERROR(cudaMemcpyToSymbol(const_conv_kernel3x3, conv_kernel3x3, sizeof(unsigned char) * 3 * 3));
		k_1D_gf_3x3_load_balance2_constant << <grid2, block >> > (gpu_input, rows, cols);
		break;
	case GAUSSIAN_3x3_vectorized16_constant:
		CHECK_CUDA_ERROR(cudaMemcpyToSymbol(const_conv_kernel3x3, conv_kernel3x3, sizeof(unsigned char) * 3 * 3));
		k_1D_gf_3x3_vectorized16_constant << <grid16, block >> > (gpu_input, rows, cols);
		break;
	case GAUSSIAN_3x3_vectorized12_constant:
		CHECK_CUDA_ERROR(cudaMemcpyToSymbol(const_conv_kernel3x3, conv_kernel3x3, sizeof(unsigned char) * 3 * 3));
		k_1D_gf_3x3_vectorized12_constant << <grid12, block >> > (gpu_input, rows, cols);
		break;
	case GAUSSIAN_3x3_vectorized8_constant:
		CHECK_CUDA_ERROR(cudaMemcpyToSymbol(const_conv_kernel3x3, conv_kernel3x3, sizeof(unsigned char) * 3 * 3));
		k_1D_gf_3x3_vectorized8_constant << <grid8, block >> > (gpu_input, rows, cols);
		break;
	case GAUSSIAN_3x3_vectorized4_constant:
		CHECK_CUDA_ERROR(cudaMemcpyToSymbol(const_conv_kernel3x3, conv_kernel3x3, sizeof(unsigned char) * 3 * 3));
		k_1D_gf_3x3_vectorized4_constant << <grid4, block >> > (gpu_input, rows, cols);
		break;
	case GAUSSIAN_3x3_vectorized2_constant:
		CHECK_CUDA_ERROR(cudaMemcpyToSymbol(const_conv_kernel3x3, conv_kernel3x3, sizeof(unsigned char) * 3 * 3));
		k_1D_gf_3x3_vectorized2_constant << <grid2, block >> > (gpu_input, rows, cols);
		break;
	case GAUSSIAN_3x3_load_balance16_shared:
		k_1D_gf_3x3_load_balance16_shared << <grid16, block >> > (gpu_input, rows, cols);
		break;
	case GAUSSIAN_3x3_load_balance12_shared:
		k_1D_gf_3x3_load_balance12_shared << <grid12, block >> > (gpu_input, rows, cols);
		break;
	case GAUSSIAN_3x3_load_balance8_shared:
		k_1D_gf_3x3_load_balance8_shared << <grid8, block >> > (gpu_input, rows, cols);
		break;
	case GAUSSIAN_3x3_load_balance4_shared:
		k_1D_gf_3x3_load_balance4_shared << <grid4, block >> > (gpu_input, rows, cols);
		break;
	case GAUSSIAN_3x3_load_balance2_shared:
		k_1D_gf_3x3_load_balance2_shared << <grid2, block >> > (gpu_input, rows, cols);
		break;
	case GAUSSIAN_3x3_vectorized16_shared:
		k_1D_gf_3x3_vectorized16_shared << <grid16, block >> > (gpu_input, rows, cols);
		break;
	case GAUSSIAN_3x3_vectorized12_shared:
		k_1D_gf_3x3_vectorized12_shared << <grid12, block >> > (gpu_input, rows, cols);
		break;
	case GAUSSIAN_3x3_vectorized8_shared:
		k_1D_gf_3x3_vectorized8_shared << <grid8, block >> > (gpu_input, rows, cols);
		break;
	case GAUSSIAN_3x3_vectorized4_shared:
		k_1D_gf_3x3_vectorized4_shared << <grid4, block >> > (gpu_input, rows, cols);
		break;
	case GAUSSIAN_3x3_vectorized2_shared:
		k_1D_gf_3x3_vectorized2_shared << <grid2, block >> > (gpu_input, rows, cols);
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