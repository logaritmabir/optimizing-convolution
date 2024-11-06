#include "kernels.cuh"
#include "utils.cuh"

inline __device__ void shift_left(unsigned char arr[][3]) {
	arr[0][0] = arr[0][1];
	arr[1][0] = arr[1][1];
	arr[2][0] = arr[2][1];
	arr[0][1] = arr[0][2];
	arr[1][1] = arr[1][2];
	arr[2][1] = arr[2][2];
}
__constant__ Npp32s NPP_Filter[9] = {1, 2, 1, 
									2, 4, 2, 
									1, 2, 1 };

__constant__ unsigned char CM_Filter[3][3] = {{1, 2, 1}, 
											{2, 4, 2}, 
											{1, 2, 1} };

__device__ unsigned char GM_Filter[3][3] = {{1, 2, 1}, 
											{2, 4, 2}, 
											{1, 2, 1} };

__global__ void GM_3x3(const unsigned char* __restrict__ input, unsigned char* __restrict__ output, const int rows, const int cols){
	int ty = blockIdx.x * blockDim.x + threadIdx.x;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;
	if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1)) {
		output[tx * cols + ty] = (GM_Filter[0][0] * input[(tx - 1) * cols + ty - 1]
		 + GM_Filter[0][1] * input[(tx - 1) * cols + ty]
		 + GM_Filter[0][2] * input[(tx - 1) * cols + ty + 1]
		 + GM_Filter[1][0] * input[tx * cols + ty - 1]
		 + GM_Filter[1][1] * input[tx * cols + ty]
		 + GM_Filter[1][2] * input[tx * cols + ty + 1]
		 + GM_Filter[2][0] * input[(tx + 1) * cols + ty - 1]
		 + GM_Filter[2][1] * input[(tx + 1) * cols + ty]
		 + GM_Filter[2][2] * input[(tx + 1) * cols + ty + 1]) >> 4;
	}
}

__global__ void CM_3x3(const unsigned char* __restrict__ input, unsigned char* __restrict__ output, const int rows, const int cols)
{
	int ty = blockIdx.x * blockDim.x + threadIdx.x;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1)) {
		output[tx * cols + ty] = (CM_Filter[0][0] * input[(tx - 1) * cols + ty - 1]
		+ CM_Filter[0][1] * input[(tx - 1) * cols + ty]
		+ CM_Filter[0][2] * input[(tx - 1) * cols + ty + 1]
		+ CM_Filter[1][0] * input[tx * cols + ty - 1]
		+ CM_Filter[1][1] * input[tx * cols + ty]
		+ CM_Filter[1][2] * input[tx * cols + ty + 1]
		+ CM_Filter[2][0] * input[(tx + 1) * cols + ty - 1]
		+ CM_Filter[2][1] * input[(tx + 1) * cols + ty]
		+ CM_Filter[2][2] * input[(tx + 1) * cols + ty + 1]) >> 4;
	}
}

__global__ void SM_3x3(const unsigned char* __restrict__ input, unsigned char* __restrict__ output, const int rows, const int cols)
{
	__shared__  unsigned char cache[34][34];

	int ty = blockIdx.x * blockDim.x + threadIdx.x;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	unsigned int cy = threadIdx.x + 1;
	unsigned int cx = threadIdx.y + 1;

    if (tx < rows && ty < cols) {
        cache[cx][cy] = input[tx * cols + ty];
    }

    if (cx == 1 && tx > 0) {
        cache[0][cy] = input[(tx - 1) * cols + ty];
    }
    if (cx == 32 && tx < rows - 1) {
        cache[33][cy] = input[(tx + 1) * cols + ty];
    }
    if (cy == 1 && ty > 0) {
        cache[cx][0] = input[tx * cols + ty - 1];
    }
    if (cy == 32 && ty < cols - 1) {
        cache[cx][33] = input[tx * cols + ty + 1];
    }
	__syncthreads();

	if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1)) {
		output[tx * cols + ty] = (GM_Filter[0][0] * cache[cx - 1][cy - 1]
		+ GM_Filter[0][1] * cache[cx - 1][cy]
		+ GM_Filter[0][2] * cache[cx - 1][cy + 1]
		+ GM_Filter[1][0] * cache[cx][cy - 1]
		+ GM_Filter[1][1] * cache[cx][cy]
		+ GM_Filter[1][2] * cache[cx][cy + 1]
		+ GM_Filter[2][0] * cache[cx + 1][cy - 1]
		+ GM_Filter[2][1] * cache[cx + 1][cy]
		+ GM_Filter[2][2] * cache[cx + 1][cy + 1]) >> 4;
	}
}

__global__ void GM_3x3_CF16(const unsigned char* __restrict__ input, unsigned char* __restrict__ output, const int rows, const int cols)
{
	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 16;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	unsigned char frame[3][3];

	if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1)) {
		frame[0][0] = input[(tx - 1) * cols + ty - 1];
		frame[0][1] = input[(tx - 1) * cols + ty];
		frame[0][2] = input[(tx - 1) * cols + ty + 1];
		frame[1][0] = input[tx * cols + ty - 1];
		frame[1][1] = input[tx * cols + ty];
		frame[1][2] = input[tx * cols + ty + 1];
		frame[2][0] = input[(tx + 1) * cols + ty - 1];
		frame[2][1] = input[(tx + 1) * cols + ty];
		frame[2][2] = input[(tx + 1) * cols + ty + 1];

		output[(tx * cols + ty)] = (GM_Filter[0][0] * frame[0][0]
		+ GM_Filter[0][1] * frame[0][1]
		+ GM_Filter[0][2] * frame[0][2]
		+ GM_Filter[1][0] * frame[1][0]
		+ GM_Filter[1][1] * frame[1][1]
		+ GM_Filter[1][2] * frame[1][2]
		+ GM_Filter[2][0] * frame[2][0]
		+ GM_Filter[2][1] * frame[2][1]
		+ GM_Filter[2][2] * frame[2][2]) >> 4;

		#pragma unroll
		for (int i = 1; i < 16; i++) {
			int _ty = ty + i;
			shift_left(frame);
			if ((tx > 0 && tx < rows - 1) && (_ty > 0 && _ty < cols - 1)) {
				frame[0][2] = input[(tx - 1) * cols + _ty + 1];
				frame[1][2] = input[tx * cols + _ty + 1];
				frame[2][2] = input[(tx + 1) * cols + _ty + 1];

				output[(tx * cols + _ty)] = (GM_Filter[0][0] * frame[0][0]
				+ GM_Filter[0][1] * frame[0][1]
				+ GM_Filter[0][2] * frame[0][2]
				+ GM_Filter[1][0] * frame[1][0]
				+ GM_Filter[1][1] * frame[1][1]
				+ GM_Filter[1][2] * frame[1][2]
				+ GM_Filter[2][0] * frame[2][0]
				+ GM_Filter[2][1] * frame[2][1]
				+ GM_Filter[2][2] * frame[2][2]) >> 4;
			}
		}
	}
}

__global__ void GM_3x3_CF12(const unsigned char* __restrict__ input, unsigned char* __restrict__ output, const int rows, const int cols)
{
	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 12;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	unsigned char frame[3][3];

	if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1)) {
		frame[0][0] = input[(tx - 1) * cols + ty - 1];
		frame[0][1] = input[(tx - 1) * cols + ty];
		frame[0][2] = input[(tx - 1) * cols + ty + 1];
		frame[1][0] = input[tx * cols + ty - 1];
		frame[1][1] = input[tx * cols + ty];
		frame[1][2] = input[tx * cols + ty + 1];
		frame[2][0] = input[(tx + 1) * cols + ty - 1];
		frame[2][1] = input[(tx + 1) * cols + ty];
		frame[2][2] = input[(tx + 1) * cols + ty + 1];

		output[(tx * cols + ty)] = (GM_Filter[0][0] * frame[0][0]
		+ GM_Filter[0][1] * frame[0][1]
		+ GM_Filter[0][2] * frame[0][2]
		+ GM_Filter[1][0] * frame[1][0]
		+ GM_Filter[1][1] * frame[1][1]
		+ GM_Filter[1][2] * frame[1][2]
		+ GM_Filter[2][0] * frame[2][0]
		+ GM_Filter[2][1] * frame[2][1]
		+ GM_Filter[2][2] * frame[2][2]) >> 4;

		#pragma unroll
		for (int i = 1; i < 12; i++) {
			int _ty = ty + i;
			shift_left(frame);
			if ((tx > 0 && tx < rows - 1) && (_ty > 0 && _ty < cols - 1)) {
				frame[0][2] = input[(tx - 1) * cols + _ty + 1];
				frame[1][2] = input[tx * cols + _ty + 1];
				frame[2][2] = input[(tx + 1) * cols + _ty + 1];

				output[(tx * cols + _ty)] = (GM_Filter[0][0] * frame[0][0]
				+ GM_Filter[0][1] * frame[0][1]
				+ GM_Filter[0][2] * frame[0][2]
				+ GM_Filter[1][0] * frame[1][0]
				+ GM_Filter[1][1] * frame[1][1]
				+ GM_Filter[1][2] * frame[1][2]
				+ GM_Filter[2][0] * frame[2][0]
				+ GM_Filter[2][1] * frame[2][1]
				+ GM_Filter[2][2] * frame[2][2]) >> 4;
			}
		}
	}
}

__global__ void GM_3x3_CF8(const unsigned char* __restrict__ input, unsigned char* __restrict__ output, const int rows, const int cols)
{
	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	unsigned char frame[3][3];

	if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1)) {
		frame[0][0] = input[(tx - 1) * cols + ty - 1];
		frame[0][1] = input[(tx - 1) * cols + ty];
		frame[0][2] = input[(tx - 1) * cols + ty + 1];
		frame[1][0] = input[tx * cols + ty - 1];
		frame[1][1] = input[tx * cols + ty];
		frame[1][2] = input[tx * cols + ty + 1];
		frame[2][0] = input[(tx + 1) * cols + ty - 1];
		frame[2][1] = input[(tx + 1) * cols + ty];
		frame[2][2] = input[(tx + 1) * cols + ty + 1];

		output[(tx * cols + ty)] = (GM_Filter[0][0] * frame[0][0]
		+ GM_Filter[0][1] * frame[0][1]
		+ GM_Filter[0][2] * frame[0][2]
		+ GM_Filter[1][0] * frame[1][0]
		+ GM_Filter[1][1] * frame[1][1]
		+ GM_Filter[1][2] * frame[1][2]
		+ GM_Filter[2][0] * frame[2][0]
		+ GM_Filter[2][1] * frame[2][1]
		+ GM_Filter[2][2] * frame[2][2]) >> 4;

		#pragma unroll
		for (int i = 1; i < 8; i++) {
			int _ty = ty + i;
			shift_left(frame);
			if ((tx > 0 && tx < rows - 1) && (_ty > 0 && _ty < cols - 1)) {
				frame[0][2] = input[(tx - 1) * cols + _ty + 1];
				frame[1][2] = input[tx * cols + _ty + 1];
				frame[2][2] = input[(tx + 1) * cols + _ty + 1];

				output[(tx * cols + _ty)] = (GM_Filter[0][0] * frame[0][0]
				+ GM_Filter[0][1] * frame[0][1]
				+ GM_Filter[0][2] * frame[0][2]
				+ GM_Filter[1][0] * frame[1][0]
				+ GM_Filter[1][1] * frame[1][1]
				+ GM_Filter[1][2] * frame[1][2]
				+ GM_Filter[2][0] * frame[2][0]
				+ GM_Filter[2][1] * frame[2][1]
				+ GM_Filter[2][2] * frame[2][2]) >> 4;
			}
		}
	}
}

__global__ void GM_3x3_CF4(const unsigned char* __restrict__ input, unsigned char* __restrict__ output, const int rows, const int cols)
{
	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned char frame[3][3];

	if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1)) {
		frame[0][0] = input[(tx - 1) * cols + ty - 1];
		frame[0][1] = input[(tx - 1) * cols + ty];
		frame[0][2] = input[(tx - 1) * cols + ty + 1];
		frame[1][0] = input[tx * cols + ty - 1];
		frame[1][1] = input[tx * cols + ty];
		frame[1][2] = input[tx * cols + ty + 1];
		frame[2][0] = input[(tx + 1) * cols + ty - 1];
		frame[2][1] = input[(tx + 1) * cols + ty];
		frame[2][2] = input[(tx + 1) * cols + ty + 1];

		output[(tx * cols + ty)] = (GM_Filter[0][0] * frame[0][0]
			+ GM_Filter[0][1] * frame[0][1]
			+ GM_Filter[0][2] * frame[0][2]
			+ GM_Filter[1][0] * frame[1][0]
			+ GM_Filter[1][1] * frame[1][1]
			+ GM_Filter[1][2] * frame[1][2]
			+ GM_Filter[2][0] * frame[2][0]
			+ GM_Filter[2][1] * frame[2][1]
			+ GM_Filter[2][2] * frame[2][2]) >> 4;

		#pragma unroll
		for (int i = 1; i < 4; i++) {
			int _ty = ty + i;
			shift_left(frame);
			if ((tx > 0 && tx < rows - 1) && (_ty > 0 && _ty < cols - 1)) {
				frame[0][2] = input[(tx - 1) * cols + _ty + 1];
				frame[1][2] = input[tx * cols + _ty + 1];
				frame[2][2] = input[(tx + 1) * cols + _ty + 1];

				output[(tx * cols + _ty)] = (GM_Filter[0][0] * frame[0][0]
					+ GM_Filter[0][1] * frame[0][1]
					+ GM_Filter[0][2] * frame[0][2]
					+ GM_Filter[1][0] * frame[1][0]
					+ GM_Filter[1][1] * frame[1][1]
					+ GM_Filter[1][2] * frame[1][2]
					+ GM_Filter[2][0] * frame[2][0]
					+ GM_Filter[2][1] * frame[2][1]
					+ GM_Filter[2][2] * frame[2][2]) >> 4;
			}
		}
	}
}

__global__ void GM_3x3_CF2(const unsigned char* __restrict__ input, unsigned char* __restrict__ output, const int rows, const int cols)
{
	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	unsigned char frame[3][3];

	if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1)) {
		frame[0][0] = input[(tx - 1) * cols + ty - 1];
		frame[0][1] = input[(tx - 1) * cols + ty];
		frame[0][2] = input[(tx - 1) * cols + ty + 1];
		frame[1][0] = input[tx * cols + ty - 1];
		frame[1][1] = input[tx * cols + ty];
		frame[1][2] = input[tx * cols + ty + 1];
		frame[2][0] = input[(tx + 1) * cols + ty - 1];
		frame[2][1] = input[(tx + 1) * cols + ty];
		frame[2][2] = input[(tx + 1) * cols + ty + 1];

		output[(tx * cols + ty)] = (GM_Filter[0][0] * frame[0][0]
		+ GM_Filter[0][1] * frame[0][1]
		+ GM_Filter[0][2] * frame[0][2]
		+ GM_Filter[1][0] * frame[1][0]
		+ GM_Filter[1][1] * frame[1][1]
		+ GM_Filter[1][2] * frame[1][2]
		+ GM_Filter[2][0] * frame[2][0]
		+ GM_Filter[2][1] * frame[2][1]
		+ GM_Filter[2][2] * frame[2][2]) >> 4;

		#pragma unroll
		for (int i = 1; i < 2; i++) {
			int _ty = ty + i;
			shift_left(frame);
			if ((tx > 0 && tx < rows - 1) && (_ty > 0 && _ty < cols - 1)) {
				frame[0][2] = input[(tx - 1) * cols + _ty + 1];
				frame[1][2] = input[tx * cols + _ty + 1];
				frame[2][2] = input[(tx + 1) * cols + _ty + 1];

				output[(tx * cols + _ty)] = (GM_Filter[0][0] * frame[0][0]
					+ GM_Filter[0][1] * frame[0][1]
					+ GM_Filter[0][2] * frame[0][2]
					+ GM_Filter[1][0] * frame[1][0]
					+ GM_Filter[1][1] * frame[1][1]
					+ GM_Filter[1][2] * frame[1][2]
					+ GM_Filter[2][0] * frame[2][0]
					+ GM_Filter[2][1] * frame[2][1]
					+ GM_Filter[2][2] * frame[2][2]) >> 4;
			}
		}
	}
}

__global__ void CM_3x3_CF16(const unsigned char* __restrict__ input, unsigned char* __restrict__ output, const int rows, const int cols)
{
	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 16;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	unsigned char frame[3][3];

	if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1)) {
		frame[0][0] = input[(tx - 1) * cols + ty - 1];
		frame[0][1] = input[(tx - 1) * cols + ty];
		frame[0][2] = input[(tx - 1) * cols + ty + 1];
		frame[1][0] = input[tx * cols + ty - 1];
		frame[1][1] = input[tx * cols + ty];
		frame[1][2] = input[tx * cols + ty + 1];
		frame[2][0] = input[(tx + 1) * cols + ty - 1];
		frame[2][1] = input[(tx + 1) * cols + ty];
		frame[2][2] = input[(tx + 1) * cols + ty + 1];

		output[(tx * cols + ty)] = (CM_Filter[0][0] * frame[0][0]
			+ CM_Filter[0][1] * frame[0][1]
			+ CM_Filter[0][2] * frame[0][2]
			+ CM_Filter[1][0] * frame[1][0]
			+ CM_Filter[1][1] * frame[1][1]
			+ CM_Filter[1][2] * frame[1][2]
			+ CM_Filter[2][0] * frame[2][0]
			+ CM_Filter[2][1] * frame[2][1]
			+ CM_Filter[2][2] * frame[2][2]) >> 4;

		#pragma unroll
		for (int i = 1; i < 16; i++) {
			int _ty = ty + i;
			shift_left(frame);
			if ((tx > 0 && tx < rows - 1) && (_ty > 0 && _ty < cols - 1)) {
				frame[0][2] = input[(tx - 1) * cols + _ty + 1];
				frame[1][2] = input[tx * cols + _ty + 1];
				frame[2][2] = input[(tx + 1) * cols + _ty + 1];

				output[(tx * cols + _ty)] = (CM_Filter[0][0] * frame[0][0]
					+ CM_Filter[0][1] * frame[0][1]
					+ CM_Filter[0][2] * frame[0][2]
					+ CM_Filter[1][0] * frame[1][0]
					+ CM_Filter[1][1] * frame[1][1]
					+ CM_Filter[1][2] * frame[1][2]
					+ CM_Filter[2][0] * frame[2][0]
					+ CM_Filter[2][1] * frame[2][1]
					+ CM_Filter[2][2] * frame[2][2]) >> 4;
			}
		}
	}
}

__global__ void CM_3x3_CF12(const unsigned char* __restrict__ input, unsigned char* __restrict__ output, const int rows, const int cols)
{
	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 12;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned char frame[3][3];

	if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1)) {
		frame[0][0] = input[(tx - 1) * cols + ty - 1];
		frame[0][1] = input[(tx - 1) * cols + ty];
		frame[0][2] = input[(tx - 1) * cols + ty + 1];
		frame[1][0] = input[tx * cols + ty - 1];
		frame[1][1] = input[tx * cols + ty];
		frame[1][2] = input[tx * cols + ty + 1];
		frame[2][0] = input[(tx + 1) * cols + ty - 1];
		frame[2][1] = input[(tx + 1) * cols + ty];
		frame[2][2] = input[(tx + 1) * cols + ty + 1];


		output[(tx * cols + ty)] = (CM_Filter[0][0] * frame[0][0]
			+ CM_Filter[0][1] * frame[0][1]
			+ CM_Filter[0][2] * frame[0][2]
			+ CM_Filter[1][0] * frame[1][0]
			+ CM_Filter[1][1] * frame[1][1]
			+ CM_Filter[1][2] * frame[1][2]
			+ CM_Filter[2][0] * frame[2][0]
			+ CM_Filter[2][1] * frame[2][1]
			+ CM_Filter[2][2] * frame[2][2]) >> 4;

		#pragma unroll
		for (int i = 1; i < 12; i++) {
			int _ty = ty + i;
			shift_left(frame);
			if ((tx > 0 && tx < rows - 1) && (_ty > 0 && _ty < cols - 1)) {
				frame[0][2] = input[(tx - 1) * cols + _ty + 1];
				frame[1][2] = input[tx * cols + _ty + 1];
				frame[2][2] = input[(tx + 1) * cols + _ty + 1];

				output[(tx * cols + _ty)] = (CM_Filter[0][0] * frame[0][0]
					+ CM_Filter[0][1] * frame[0][1]
					+ CM_Filter[0][2] * frame[0][2]
					+ CM_Filter[1][0] * frame[1][0]
					+ CM_Filter[1][1] * frame[1][1]
					+ CM_Filter[1][2] * frame[1][2]
					+ CM_Filter[2][0] * frame[2][0]
					+ CM_Filter[2][1] * frame[2][1]
					+ CM_Filter[2][2] * frame[2][2]) >> 4;
			}
		}
	}
}

__global__ void CM_3x3_CF8(const unsigned char* __restrict__ input, unsigned char* __restrict__ output, const int rows, const int cols)
{
	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned char frame[3][3];

	if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1)) {
		frame[0][0] = input[(tx - 1) * cols + ty - 1];
		frame[0][1] = input[(tx - 1) * cols + ty];
		frame[0][2] = input[(tx - 1) * cols + ty + 1];
		frame[1][0] = input[tx * cols + ty - 1];
		frame[1][1] = input[tx * cols + ty];
		frame[1][2] = input[tx * cols + ty + 1];
		frame[2][0] = input[(tx + 1) * cols + ty - 1];
		frame[2][1] = input[(tx + 1) * cols + ty];
		frame[2][2] = input[(tx + 1) * cols + ty + 1];

		output[(tx * cols + ty)] = (CM_Filter[0][0] * frame[0][0]
			+ CM_Filter[0][1] * frame[0][1]
			+ CM_Filter[0][2] * frame[0][2]
			+ CM_Filter[1][0] * frame[1][0]
			+ CM_Filter[1][1] * frame[1][1]
			+ CM_Filter[1][2] * frame[1][2]
			+ CM_Filter[2][0] * frame[2][0]
			+ CM_Filter[2][1] * frame[2][1]
			+ CM_Filter[2][2] * frame[2][2]) >> 4;

		#pragma unroll
		for (int i = 1; i < 8; i++) {
			int _ty = ty + i;
			shift_left(frame);
			if ((tx > 0 && tx < rows - 1) && (_ty > 0 && _ty < cols - 1)) {
				frame[0][2] = input[(tx - 1) * cols + _ty + 1];
				frame[1][2] = input[tx * cols + _ty + 1];
				frame[2][2] = input[(tx + 1) * cols + _ty + 1];

				output[(tx * cols + _ty)] = (CM_Filter[0][0] * frame[0][0]
					+ CM_Filter[0][1] * frame[0][1]
					+ CM_Filter[0][2] * frame[0][2]
					+ CM_Filter[1][0] * frame[1][0]
					+ CM_Filter[1][1] * frame[1][1]
					+ CM_Filter[1][2] * frame[1][2]
					+ CM_Filter[2][0] * frame[2][0]
					+ CM_Filter[2][1] * frame[2][1]
					+ CM_Filter[2][2] * frame[2][2]) >> 4;
			}
		}
	}
}

__global__ void CM_3x3_CF4(const unsigned char* __restrict__ input, unsigned char* __restrict__ output, const int rows, const int cols)
{
	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	unsigned char frame[3][3];

	if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1)) {
		frame[0][0] = input[(tx - 1) * cols + ty - 1];
		frame[0][1] = input[(tx - 1) * cols + ty];
		frame[0][2] = input[(tx - 1) * cols + ty + 1];
		frame[1][0] = input[tx * cols + ty - 1];
		frame[1][1] = input[tx * cols + ty];
		frame[1][2] = input[tx * cols + ty + 1];
		frame[2][0] = input[(tx + 1) * cols + ty - 1];
		frame[2][1] = input[(tx + 1) * cols + ty];
		frame[2][2] = input[(tx + 1) * cols + ty + 1];

		output[(tx * cols + ty)] = (CM_Filter[0][0] * frame[0][0]
			+ CM_Filter[0][1] * frame[0][1]
			+ CM_Filter[0][2] * frame[0][2]
			+ CM_Filter[1][0] * frame[1][0]
			+ CM_Filter[1][1] * frame[1][1]
			+ CM_Filter[1][2] * frame[1][2]
			+ CM_Filter[2][0] * frame[2][0]
			+ CM_Filter[2][1] * frame[2][1]
			+ CM_Filter[2][2] * frame[2][2]) >> 4;

		#pragma unroll
		for (int i = 1; i < 4; i++) {
			int _ty = ty + i;
			shift_left(frame);
			if ((tx > 0 && tx < rows - 1) && (_ty > 0 && _ty < cols - 1)) {
				frame[0][2] = input[(tx - 1) * cols + _ty + 1];
				frame[1][2] = input[tx * cols + _ty + 1];
				frame[2][2] = input[(tx + 1) * cols + _ty + 1];

				output[(tx * cols + _ty)] = (CM_Filter[0][0] * frame[0][0]
					+ CM_Filter[0][1] * frame[0][1]
					+ CM_Filter[0][2] * frame[0][2]
					+ CM_Filter[1][0] * frame[1][0]
					+ CM_Filter[1][1] * frame[1][1]
					+ CM_Filter[1][2] * frame[1][2]
					+ CM_Filter[2][0] * frame[2][0]
					+ CM_Filter[2][1] * frame[2][1]
					+ CM_Filter[2][2] * frame[2][2]) >> 4;
			}
		}
	}
}

__global__ void CM_3x3_CF2(const unsigned char* __restrict__ input, unsigned char* __restrict__ output, const int rows, const int cols)
{
	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	unsigned char frame[3][3];

	if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1)) {
		frame[0][0] = input[(tx - 1) * cols + ty - 1];
		frame[0][1] = input[(tx - 1) * cols + ty];
		frame[0][2] = input[(tx - 1) * cols + ty + 1];
		frame[1][0] = input[tx * cols + ty - 1];
		frame[1][1] = input[tx * cols + ty];
		frame[1][2] = input[tx * cols + ty + 1];
		frame[2][0] = input[(tx + 1) * cols + ty - 1];
		frame[2][1] = input[(tx + 1) * cols + ty];
		frame[2][2] = input[(tx + 1) * cols + ty + 1];

		output[(tx * cols + ty)] = (CM_Filter[0][0] * frame[0][0]
			+ CM_Filter[0][1] * frame[0][1]
			+ CM_Filter[0][2] * frame[0][2]
			+ CM_Filter[1][0] * frame[1][0]
			+ CM_Filter[1][1] * frame[1][1]
			+ CM_Filter[1][2] * frame[1][2]
			+ CM_Filter[2][0] * frame[2][0]
			+ CM_Filter[2][1] * frame[2][1]
			+ CM_Filter[2][2] * frame[2][2]) >> 4;

		#pragma unroll
		for (int i = 1; i < 2; i++) {
			int _ty = ty + i;
			shift_left(frame);
			if ((tx > 0 && tx < rows - 1) && (_ty > 0 && _ty < cols - 1)) {
				frame[0][2] = input[(tx - 1) * cols + _ty + 1];
				frame[1][2] = input[tx * cols + _ty + 1];
				frame[2][2] = input[(tx + 1) * cols + _ty + 1];

				output[(tx * cols + _ty)] = (CM_Filter[0][0] * frame[0][0]
					+ CM_Filter[0][1] * frame[0][1]
					+ CM_Filter[0][2] * frame[0][2]
					+ CM_Filter[1][0] * frame[1][0]
					+ CM_Filter[1][1] * frame[1][1]
					+ CM_Filter[1][2] * frame[1][2]
					+ CM_Filter[2][0] * frame[2][0]
					+ CM_Filter[2][1] * frame[2][1]
					+ CM_Filter[2][2] * frame[2][2]) >> 4;
			}
		}
	}
}

__global__ void SM_3x3_CF16(const unsigned char* __restrict__ input, unsigned char* __restrict__ output, const int rows, const int cols)
{
	__shared__  unsigned char cache[34][514];

	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 16;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	int cy = threadIdx.x * 16 + 1;
	int cx = threadIdx.y + 1;

	if ((tx > 0 && tx < rows - 1) && (ty < cols - 1)) {
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

		unsigned char frame[3][3];

		frame[0][0] = cache[cx - 1][cy - 1];
		frame[0][1] = cache[cx - 1][cy];
		frame[0][2] = cache[cx - 1][cy + 1];
		frame[1][0] = cache[cx][cy - 1];
		frame[1][1] = cache[cx][cy];
		frame[1][2] = cache[cx][cy + 1];
		frame[2][0] = cache[cx + 1][cy - 1];
		frame[2][1] = cache[cx + 1][cy];
		frame[2][2] = cache[cx + 1][cy + 1];

		output[tx * cols + ty] = (GM_Filter[0][0] * frame[0][0]
		+ GM_Filter[0][1] * frame[0][1]
		+ GM_Filter[0][2] * frame[0][2]
		+ GM_Filter[1][0] * frame[1][0]
		+ GM_Filter[1][1] * frame[1][1]
		+ GM_Filter[1][2] * frame[1][2]
		+ GM_Filter[2][0] * frame[2][0]
		+ GM_Filter[2][1] * frame[2][1]
		+ GM_Filter[2][2] * frame[2][2]) >> 4;

		#pragma unroll
		for (int i = 1; i < 16; i++) {
			int _ty = ty + i;
			int _cy = cy + i;
			shift_left(frame);
			frame[0][2] = cache[cx - 1][_cy + 1];
			frame[1][2] = cache[cx][_cy + 1];
			frame[2][2] = cache[cx + 1][_cy + 1];

			if (_ty < cols - 1) {
				output[tx * cols + _ty] = (GM_Filter[0][0] * frame[0][0]
				+ GM_Filter[0][1] * frame[0][1]
				+ GM_Filter[0][2] * frame[0][2]
				+ GM_Filter[1][0] * frame[1][0]
				+ GM_Filter[1][1] * frame[1][1]
				+ GM_Filter[1][2] * frame[1][2]
				+ GM_Filter[2][0] * frame[2][0]
				+ GM_Filter[2][1] * frame[2][1]
				+ GM_Filter[2][2] * frame[2][2]) >> 4;
			}
		}
	}
}

__global__ void SM_3x3_CF12(const unsigned char* __restrict__ input, unsigned char *__restrict__ output, const int rows, const int cols)
{
	__shared__  unsigned char cache[34][386];

	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 12;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	int cy = threadIdx.x * 12 + 1;
	int cx = threadIdx.y + 1;

	if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1)) {
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
		if (cx == 1) {
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

		unsigned char frame[3][3];

		frame[0][0] = cache[cx - 1][cy - 1];
		frame[0][1] = cache[cx - 1][cy];
		frame[0][2] = cache[cx - 1][cy + 1];
		frame[1][0] = cache[cx][cy - 1];
		frame[1][1] = cache[cx][cy];
		frame[1][2] = cache[cx][cy + 1];
		frame[2][0] = cache[cx + 1][cy - 1];
		frame[2][1] = cache[cx + 1][cy];
		frame[2][2] = cache[cx + 1][cy + 1];

		output[tx * cols + ty] = (GM_Filter[0][0] * frame[0][0]
		+ GM_Filter[0][1] * frame[0][1]
		+ GM_Filter[0][2] * frame[0][2]
		+ GM_Filter[1][0] * frame[1][0]
		+ GM_Filter[1][1] * frame[1][1]
		+ GM_Filter[1][2] * frame[1][2]
		+ GM_Filter[2][0] * frame[2][0]
		+ GM_Filter[2][1] * frame[2][1]
		+ GM_Filter[2][2] * frame[2][2]) >> 4;

		#pragma unroll
		for (int i = 1; i < 12; i++) {
			int _ty = ty + i;
			int _cy = cy + i;
			shift_left(frame);
			frame[0][2] = cache[cx - 1][_cy + 1];
			frame[1][2] = cache[cx][_cy + 1];
			frame[2][2] = cache[cx + 1][_cy + 1];

			if (_ty < cols - 1) {
				output[tx * cols + _ty] = (GM_Filter[0][0] * frame[0][0]
				+ GM_Filter[0][1] * frame[0][1]
				+ GM_Filter[0][2] * frame[0][2]
				+ GM_Filter[1][0] * frame[1][0]
				+ GM_Filter[1][1] * frame[1][1]
				+ GM_Filter[1][2] * frame[1][2]
				+ GM_Filter[2][0] * frame[2][0]
				+ GM_Filter[2][1] * frame[2][1]
				+ GM_Filter[2][2] * frame[2][2]) >> 4;
			}
		}
	}
}

__global__ void SM_3x3_CF8(const unsigned char* __restrict__ input, unsigned char* __restrict__ output, const int rows, const int cols)
{
	__shared__  unsigned char cache[34][258];

	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	int cy = threadIdx.x * 8 + 1;
	int cx = threadIdx.y + 1;

	if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1)) {
		cache[cx][cy] = input[tx * cols + ty];
		cache[cx][cy + 1] = input[tx * cols + ty + 1];
		cache[cx][cy + 2] = input[tx * cols + ty + 2];
		cache[cx][cy + 3] = input[tx * cols + ty + 3];
		cache[cx][cy + 4] = input[tx * cols + ty + 4];
		cache[cx][cy + 5] = input[tx * cols + ty + 5];
		cache[cx][cy + 6] = input[tx * cols + ty + 6];
		cache[cx][cy + 7] = input[tx * cols + ty + 7];
		if (cx == 1) { 
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

		unsigned char frame[3][3];

		frame[0][0] = cache[cx - 1][cy - 1];
		frame[0][1] = cache[cx - 1][cy];
		frame[0][2] = cache[cx - 1][cy + 1];
		frame[1][0] = cache[cx][cy - 1];
		frame[1][1] = cache[cx][cy];
		frame[1][2] = cache[cx][cy + 1];
		frame[2][0] = cache[cx + 1][cy - 1];
		frame[2][1] = cache[cx + 1][cy];
		frame[2][2] = cache[cx + 1][cy + 1];

		output[tx * cols + ty] = (GM_Filter[0][0] * frame[0][0]
		+ GM_Filter[0][1] * frame[0][1]
		+ GM_Filter[0][2] * frame[0][2]
		+ GM_Filter[1][0] * frame[1][0]
		+ GM_Filter[1][1] * frame[1][1]
		+ GM_Filter[1][2] * frame[1][2]
		+ GM_Filter[2][0] * frame[2][0]
		+ GM_Filter[2][1] * frame[2][1]
		+ GM_Filter[2][2] * frame[2][2]) >> 4;

		#pragma unroll
		for (int i = 1; i < 8; i++) {
			int _ty = ty + i;
			int _cy = cy + i;
			shift_left(frame);
			frame[0][2] = cache[cx - 1][_cy + 1];
			frame[1][2] = cache[cx][_cy + 1];
			frame[2][2] = cache[cx + 1][_cy + 1];

			if (_ty < cols - 1) {
				output[tx * cols + _ty] = (GM_Filter[0][0] * frame[0][0]
				+ GM_Filter[0][1] * frame[0][1]
				+ GM_Filter[0][2] * frame[0][2]
				+ GM_Filter[1][0] * frame[1][0]
				+ GM_Filter[1][1] * frame[1][1]
				+ GM_Filter[1][2] * frame[1][2]
				+ GM_Filter[2][0] * frame[2][0]
				+ GM_Filter[2][1] * frame[2][1]
				+ GM_Filter[2][2] * frame[2][2]) >> 4;
			}
		}
	}
}

__global__ void SM_3x3_CF4(const unsigned char* __restrict__ input, unsigned char* __restrict__ output, const int rows, const int cols)
{
	__shared__  unsigned char cache[34][130];

	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	int cy = threadIdx.x * 4 + 1;
	int cx = threadIdx.y + 1;

	if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1)) {
	cache[cx][cy] = input[tx * cols + ty];
	cache[cx][cy + 1] = input[tx * cols + ty + 1];
	cache[cx][cy + 2] = input[tx * cols + ty + 2];
	cache[cx][cy + 3] = input[tx * cols + ty + 3];
		
		if (cx == 1) {
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

		unsigned char frame[3][3];

		frame[0][0] = cache[cx - 1][cy - 1];
		frame[0][1] = cache[cx - 1][cy];
		frame[0][2] = cache[cx - 1][cy + 1];
		frame[1][0] = cache[cx][cy - 1];
		frame[1][1] = cache[cx][cy];
		frame[1][2] = cache[cx][cy + 1];
		frame[2][0] = cache[cx + 1][cy - 1];
		frame[2][1] = cache[cx + 1][cy];
		frame[2][2] = cache[cx + 1][cy + 1];

		output[tx * cols + ty] = (GM_Filter[0][0] * frame[0][0]
		+ GM_Filter[0][1] * frame[0][1]
		+ GM_Filter[0][2] * frame[0][2]
		+ GM_Filter[1][0] * frame[1][0]
		+ GM_Filter[1][1] * frame[1][1]
		+ GM_Filter[1][2] * frame[1][2]
		+ GM_Filter[2][0] * frame[2][0]
		+ GM_Filter[2][1] * frame[2][1]
		+ GM_Filter[2][2] * frame[2][2]) >> 4;

		#pragma unroll
		for (int i = 1; i < 4; i++) {
			int _ty = ty + i;
			int _cy = cy + i;
			shift_left(frame);
			frame[0][2] = cache[cx - 1][_cy + 1];
			frame[1][2] = cache[cx][_cy + 1];
			frame[2][2] = cache[cx + 1][_cy + 1];

			if (_ty < cols - 1) {
				output[tx * cols + _ty] = (GM_Filter[0][0] * frame[0][0]
				+ GM_Filter[0][1] * frame[0][1]
				+ GM_Filter[0][2] * frame[0][2]
				+ GM_Filter[1][0] * frame[1][0]
				+ GM_Filter[1][1] * frame[1][1]
				+ GM_Filter[1][2] * frame[1][2]
				+ GM_Filter[2][0] * frame[2][0]
				+ GM_Filter[2][1] * frame[2][1]
				+ GM_Filter[2][2] * frame[2][2]) >> 4;
			}
		}
	}
}

__global__ void SM_3x3_CF2(const unsigned char* __restrict__ input, unsigned char* __restrict__ output, const int rows, const int cols)
{
	__shared__  unsigned char cache[34][66];

	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	int cy = threadIdx.x * 2 + 1;
	int cx = threadIdx.y + 1;

	if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1)) {
		cache[cx][cy] = input[tx * cols + ty];
		cache[cx][cy + 1] = input[tx * cols + ty + 1];

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

		unsigned char frame[3][3];

		frame[0][0] = cache[cx - 1][cy - 1];
		frame[0][1] = cache[cx - 1][cy];
		frame[0][2] = cache[cx - 1][cy + 1];
		frame[1][0] = cache[cx][cy - 1];
		frame[1][1] = cache[cx][cy];
		frame[1][2] = cache[cx][cy + 1];
		frame[2][0] = cache[cx + 1][cy - 1];
		frame[2][1] = cache[cx + 1][cy];
		frame[2][2] = cache[cx + 1][cy + 1];

		output[tx * cols + ty] = (GM_Filter[0][0] * frame[0][0]
		+ GM_Filter[0][1] * frame[0][1]
		+ GM_Filter[0][2] * frame[0][2]
		+ GM_Filter[1][0] * frame[1][0]
		+ GM_Filter[1][1] * frame[1][1]
		+ GM_Filter[1][2] * frame[1][2]
		+ GM_Filter[2][0] * frame[2][0]
		+ GM_Filter[2][1] * frame[2][1]
		+ GM_Filter[2][2] * frame[2][2]) >> 4;

		#pragma unroll
		for (int i = 1; i < 2; i++) {
			int _ty = ty + i;
			int _cy = cy + i;
			shift_left(frame);
			frame[0][2] = cache[cx - 1][_cy + 1];
			frame[1][2] = cache[cx][_cy + 1];
			frame[2][2] = cache[cx + 1][_cy + 1];

			if ( _ty < cols - 1) {
				output[tx * cols + _ty] = (GM_Filter[0][0] * frame[0][0]
				+ GM_Filter[0][1] * frame[0][1]
				+ GM_Filter[0][2] * frame[0][2]
				+ GM_Filter[1][0] * frame[1][0]
				+ GM_Filter[1][1] * frame[1][1]
				+ GM_Filter[1][2] * frame[1][2]
				+ GM_Filter[2][0] * frame[2][0]
				+ GM_Filter[2][1] * frame[2][1]
				+ GM_Filter[2][2] * frame[2][2]) >> 4;
			}
		}
	}
}

__global__ void GM_3x3_CF16_Vec(const unsigned char* __restrict__ input, unsigned char* __restrict__ output, const int rows, const int cols)
{
	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 16;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	int vals[16] = { 0 };
	unsigned char frame[3][3];

	if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty  < cols - 1)) {
		frame[0][0] = input[(tx - 1) * cols + ty - 1];
		frame[0][1] = input[(tx - 1) * cols + ty];
		frame[0][2] = input[(tx - 1) * cols + ty + 1];
		frame[1][0] = input[tx * cols + ty - 1];
		frame[1][1] = input[tx * cols + ty];
		frame[1][2] = input[tx * cols + ty + 1];
		frame[2][0] = input[(tx + 1) * cols + ty - 1];
		frame[2][1] = input[(tx + 1) * cols + ty];
		frame[2][2] = input[(tx + 1) * cols + ty + 1];

		vals[0] = (GM_Filter[0][0] * frame[0][0]
			+ GM_Filter[0][1] * frame[0][1]
			+ GM_Filter[0][2] * frame[0][2]
			+ GM_Filter[1][0] * frame[1][0]
			+ GM_Filter[1][1] * frame[1][1]
			+ GM_Filter[1][2] * frame[1][2]
			+ GM_Filter[2][0] * frame[2][0]
			+ GM_Filter[2][1] * frame[2][1]
			+ GM_Filter[2][2] * frame[2][2]) >> 4;

		#pragma unroll
		for (int i = 1; i < 16; i++) {
			int _ty = ty + i;
			shift_left(frame);
			if ((tx > 0 && tx < rows - 1) && (_ty > 0 && _ty < cols - 1)) {
				frame[0][2] = input[(tx - 1) * cols + _ty + 1];
				frame[1][2] = input[tx * cols + _ty + 1];
				frame[2][2] = input[(tx + 1) * cols + _ty + 1];

				vals[i] = (GM_Filter[0][0] * frame[0][0]
					+ GM_Filter[0][1] * frame[0][1]
					+ GM_Filter[0][2] * frame[0][2]
					+ GM_Filter[1][0] * frame[1][0]
					+ GM_Filter[1][1] * frame[1][1]
					+ GM_Filter[1][2] * frame[1][2]
					+ GM_Filter[2][0] * frame[2][0]
					+ GM_Filter[2][1] * frame[2][1]
					+ GM_Filter[2][2] * frame[2][2]) >> 4;
			}
		}
		reinterpret_cast<uchar4*>(&output[(tx * cols + ty)])[0] = make_uchar4(vals[0], vals[1], vals[2], vals[3]);
		reinterpret_cast<uchar4*>(&output[(tx * cols + ty + 4)])[0] = make_uchar4(vals[4], vals[5], vals[6], vals[7]);
		reinterpret_cast<uchar4*>(&output[(tx * cols + ty + 8)])[0] = make_uchar4(vals[8], vals[9], vals[10], vals[11]);
		reinterpret_cast<uchar4*>(&output[(tx * cols + ty + 12)])[0] = make_uchar4(vals[12], vals[13], vals[14], vals[15]);
	}
}

__global__ void GM_3x3_CF12_Vec(const unsigned char* __restrict__ input, unsigned char* __restrict__ output, const int rows, const int cols)
{
	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 12;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	int vals[12] = { 0 };
	unsigned char frame[3][3];

	if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1)) {
		frame[0][0] = input[(tx - 1) * cols + ty - 1];
		frame[0][1] = input[(tx - 1) * cols + ty];
		frame[0][2] = input[(tx - 1) * cols + ty + 1];
		frame[1][0] = input[tx * cols + ty - 1];
		frame[1][1] = input[tx * cols + ty];
		frame[1][2] = input[tx * cols + ty + 1];
		frame[2][0] = input[(tx + 1) * cols + ty - 1];
		frame[2][1] = input[(tx + 1) * cols + ty];
		frame[2][2] = input[(tx + 1) * cols + ty + 1];

		vals[0] = (GM_Filter[0][0] * frame[0][0]
			+ GM_Filter[0][1] * frame[0][1]
			+ GM_Filter[0][2] * frame[0][2]
			+ GM_Filter[1][0] * frame[1][0]
			+ GM_Filter[1][1] * frame[1][1]
			+ GM_Filter[1][2] * frame[1][2]
			+ GM_Filter[2][0] * frame[2][0]
			+ GM_Filter[2][1] * frame[2][1]
			+ GM_Filter[2][2] * frame[2][2]) >> 4;

		#pragma unroll
		for (int i = 1; i < 12; i++) {
			int _ty = ty + i;
			shift_left(frame);
			if ((tx > 0 && tx < rows - 1) && (_ty > 0 && _ty < cols - 1)) {
				frame[0][2] = input[(tx - 1) * cols + _ty + 1];
				frame[1][2] = input[tx * cols + _ty + 1];
				frame[2][2] = input[(tx + 1) * cols + _ty + 1];

				vals[i] = (GM_Filter[0][0] * frame[0][0]
					+ GM_Filter[0][1] * frame[0][1]
					+ GM_Filter[0][2] * frame[0][2]
					+ GM_Filter[1][0] * frame[1][0]
					+ GM_Filter[1][1] * frame[1][1]
					+ GM_Filter[1][2] * frame[1][2]
					+ GM_Filter[2][0] * frame[2][0]
					+ GM_Filter[2][1] * frame[2][1]
					+ GM_Filter[2][2] * frame[2][2]) >> 4;
			}
		}
		reinterpret_cast<uchar4*>(&output[(tx * cols + ty)])[0] = make_uchar4(vals[0], vals[1], vals[2], vals[3]);
		reinterpret_cast<uchar4*>(&output[(tx * cols + ty + 4)])[0] = make_uchar4(vals[4], vals[5], vals[6], vals[7]);
		reinterpret_cast<uchar4*>(&output[(tx * cols + ty + 8)])[0] = make_uchar4(vals[8], vals[9], vals[10], vals[11]);
	}
}
__global__ void GM_3x3_CF8_Vec(const unsigned char* __restrict__ input, unsigned char* __restrict__ output, const int rows, const int cols)
{
	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	int vals[8] = { 0 };
	unsigned char frame[3][3];

	if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1)) {
		frame[0][0] = input[(tx - 1) * cols + ty - 1];
		frame[0][1] = input[(tx - 1) * cols + ty];
		frame[0][2] = input[(tx - 1) * cols + ty + 1];
		frame[1][0] = input[tx * cols + ty - 1];
		frame[1][1] = input[tx * cols + ty];
		frame[1][2] = input[tx * cols + ty + 1];
		frame[2][0] = input[(tx + 1) * cols + ty - 1];
		frame[2][1] = input[(tx + 1) * cols + ty];
		frame[2][2] = input[(tx + 1) * cols + ty + 1];

		vals[0] = (GM_Filter[0][0] * frame[0][0]
			+ GM_Filter[0][1] * frame[0][1]
			+ GM_Filter[0][2] * frame[0][2]
			+ GM_Filter[1][0] * frame[1][0]
			+ GM_Filter[1][1] * frame[1][1]
			+ GM_Filter[1][2] * frame[1][2]
			+ GM_Filter[2][0] * frame[2][0]
			+ GM_Filter[2][1] * frame[2][1]
			+ GM_Filter[2][2] * frame[2][2]) >> 4;

		#pragma unroll
		for (int i = 1; i < 8; i++) {
			int _ty = ty + i;
			shift_left(frame);
			if ((tx > 0 && tx < rows - 1) && (_ty > 0 && _ty < cols - 1)) {
				frame[0][2] = input[(tx - 1) * cols + _ty + 1];
				frame[1][2] = input[tx * cols + _ty + 1];
				frame[2][2] = input[(tx + 1) * cols + _ty + 1];

				vals[i] = (GM_Filter[0][0] * frame[0][0]
					+ GM_Filter[0][1] * frame[0][1]
					+ GM_Filter[0][2] * frame[0][2]
					+ GM_Filter[1][0] * frame[1][0]
					+ GM_Filter[1][1] * frame[1][1]
					+ GM_Filter[1][2] * frame[1][2]
					+ GM_Filter[2][0] * frame[2][0]
					+ GM_Filter[2][1] * frame[2][1]
					+ GM_Filter[2][2] * frame[2][2]) >> 4;
			}
		}
		reinterpret_cast<uchar4*>(&output[(tx * cols + ty)])[0] = make_uchar4(vals[0], vals[1], vals[2], vals[3]);
		reinterpret_cast<uchar4*>(&output[(tx * cols + ty + 4)])[0] = make_uchar4(vals[4], vals[5], vals[6], vals[7]);
	}
}

__global__ void GM_3x3_CF4_Vec(const unsigned char* __restrict__ input, unsigned char* __restrict__ output, const int rows, const int cols)
{
	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	int vals[4] = { 0 };

	unsigned char frame[3][3];

	if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1)) {
		frame[0][0] = input[(tx - 1) * cols + ty - 1];
		frame[0][1] = input[(tx - 1) * cols + ty];
		frame[0][2] = input[(tx - 1) * cols + ty + 1];
		frame[1][0] = input[tx * cols + ty - 1];
		frame[1][1] = input[tx * cols + ty];
		frame[1][2] = input[tx * cols + ty + 1];
		frame[2][0] = input[(tx + 1) * cols + ty - 1];
		frame[2][1] = input[(tx + 1) * cols + ty];
		frame[2][2] = input[(tx + 1) * cols + ty + 1];

		vals[0] = (GM_Filter[0][0] * frame[0][0]
			+ GM_Filter[0][1] * frame[0][1]
			+ GM_Filter[0][2] * frame[0][2]
			+ GM_Filter[1][0] * frame[1][0]
			+ GM_Filter[1][1] * frame[1][1]
			+ GM_Filter[1][2] * frame[1][2]
			+ GM_Filter[2][0] * frame[2][0]
			+ GM_Filter[2][1] * frame[2][1]
			+ GM_Filter[2][2] * frame[2][2]) >> 4; 

		#pragma unroll
		for (int i = 1; i < 4; i++) {
			int _ty = ty + i;
			shift_left(frame);
			if ((tx > 0 && tx < rows - 1) && (_ty > 0 && _ty < cols - 1)) {
				frame[0][2] = input[(tx - 1) * cols + _ty + 1];
				frame[1][2] = input[tx * cols + _ty + 1];
				frame[2][2] = input[(tx + 1) * cols + _ty + 1];

				vals[i] = (GM_Filter[0][0] * frame[0][0]
					+ GM_Filter[0][1] * frame[0][1]
					+ GM_Filter[0][2] * frame[0][2]
					+ GM_Filter[1][0] * frame[1][0]
					+ GM_Filter[1][1] * frame[1][1]
					+ GM_Filter[1][2] * frame[1][2]
					+ GM_Filter[2][0] * frame[2][0]
					+ GM_Filter[2][1] * frame[2][1]
					+ GM_Filter[2][2] * frame[2][2]) >> 4;
			}
		}
		reinterpret_cast<uchar4*>(&output[(tx * cols + ty)])[0] = make_uchar4(vals[0], vals[1], vals[2], vals[3]);
	}
}

__global__ void GM_3x3_CF2_Vec(const unsigned char* __restrict__ input, unsigned char* __restrict__ output, const int rows, const int cols)
{
	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	int vals[2] = { 0 };

	unsigned char frame[3][3];

	if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1)) {
		frame[0][0] = input[(tx - 1) * cols + ty - 1];
		frame[0][1] = input[(tx - 1) * cols + ty];
		frame[0][2] = input[(tx - 1) * cols + ty + 1];
		frame[1][0] = input[tx * cols + ty - 1];
		frame[1][1] = input[tx * cols + ty];
		frame[1][2] = input[tx * cols + ty + 1];
		frame[2][0] = input[(tx + 1) * cols + ty - 1];
		frame[2][1] = input[(tx + 1) * cols + ty];
		frame[2][2] = input[(tx + 1) * cols + ty + 1];

		vals[0] = (GM_Filter[0][0] * frame[0][0]
			+ GM_Filter[0][1] * frame[0][1]
			+ GM_Filter[0][2] * frame[0][2]
			+ GM_Filter[1][0] * frame[1][0]
			+ GM_Filter[1][1] * frame[1][1]
			+ GM_Filter[1][2] * frame[1][2]
			+ GM_Filter[2][0] * frame[2][0]
			+ GM_Filter[2][1] * frame[2][1]
			+ GM_Filter[2][2] * frame[2][2]) >> 4;

		#pragma unroll
		for (int i = 1; i < 2; i++) {
			int _ty = ty + i;
			shift_left(frame);
			if ((tx > 0 && tx < rows - 1) && (_ty > 0 && _ty < cols - 1)) {
				frame[0][2] = input[(tx - 1) * cols + _ty + 1];
				frame[1][2] = input[tx * cols + _ty + 1];
				frame[2][2] = input[(tx + 1) * cols + _ty + 1];

				vals[i] = (GM_Filter[0][0] * frame[0][0]
					+ GM_Filter[0][1] * frame[0][1]
					+ GM_Filter[0][2] * frame[0][2]
					+ GM_Filter[1][0] * frame[1][0]
					+ GM_Filter[1][1] * frame[1][1]
					+ GM_Filter[1][2] * frame[1][2]
					+ GM_Filter[2][0] * frame[2][0]
					+ GM_Filter[2][1] * frame[2][1]
					+ GM_Filter[2][2] * frame[2][2]) >> 4;
			}
		}
		reinterpret_cast<uchar2*>(&output[(tx * cols + ty)])[0] = make_uchar2(vals[0], vals[1]);
	}
}

__global__ void CM_3x3_CF16_Vec(const unsigned char* __restrict__ input, unsigned char* __restrict__ output, const int rows, const int cols)
{
	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 16;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	int vals[16] = { 0 };
	unsigned char frame[3][3];

	if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1)) {
		frame[0][0] = input[(tx - 1) * cols + ty - 1];
		frame[0][1] = input[(tx - 1) * cols + ty];
		frame[0][2] = input[(tx - 1) * cols + ty + 1];
		frame[1][0] = input[tx * cols + ty - 1];
		frame[1][1] = input[tx * cols + ty];
		frame[1][2] = input[tx * cols + ty + 1];
		frame[2][0] = input[(tx + 1) * cols + ty - 1];
		frame[2][1] = input[(tx + 1) * cols + ty];
		frame[2][2] = input[(tx + 1) * cols + ty + 1];

		vals[0] = (CM_Filter[0][0] * frame[0][0]
			+ CM_Filter[0][1] * frame[0][1]
			+ CM_Filter[0][2] * frame[0][2]
			+ CM_Filter[1][0] * frame[1][0]
			+ CM_Filter[1][1] * frame[1][1]
			+ CM_Filter[1][2] * frame[1][2]
			+ CM_Filter[2][0] * frame[2][0]
			+ CM_Filter[2][1] * frame[2][1]
			+ CM_Filter[2][2] * frame[2][2]) >> 4;

		#pragma unroll
		for (int i = 1; i < 16; i++) {
			int _ty = ty + i;
			shift_left(frame);
			if ((tx > 0 && tx < rows - 1) && (_ty > 0 && _ty < cols - 1)) {
				frame[0][2] = input[(tx - 1) * cols + _ty + 1];
				frame[1][2] = input[tx * cols + _ty + 1];
				frame[2][2] = input[(tx + 1) * cols + _ty + 1];

				vals[i] = (CM_Filter[0][0] * frame[0][0]
					+ CM_Filter[0][1] * frame[0][1]
					+ CM_Filter[0][2] * frame[0][2]
					+ CM_Filter[1][0] * frame[1][0]
					+ CM_Filter[1][1] * frame[1][1]
					+ CM_Filter[1][2] * frame[1][2]
					+ CM_Filter[2][0] * frame[2][0]
					+ CM_Filter[2][1] * frame[2][1]
					+ CM_Filter[2][2] * frame[2][2]) >> 4;
			}
		}
		reinterpret_cast<uchar4*>(&output[(tx * cols + ty)])[0] = make_uchar4(vals[0], vals[1], vals[2], vals[3]);
		reinterpret_cast<uchar4*>(&output[(tx * cols + ty + 4)])[0] = make_uchar4(vals[4], vals[5], vals[6], vals[7]);
		reinterpret_cast<uchar4*>(&output[(tx * cols + ty + 8)])[0] = make_uchar4(vals[8], vals[9], vals[10], vals[11]);
		reinterpret_cast<uchar4*>(&output[(tx * cols + ty + 12)])[0] = make_uchar4(vals[12], vals[13], vals[14], vals[15]);
	}
}

__global__ void CM_3x3_CF12_Vec(const unsigned char* __restrict__ input, unsigned char* __restrict__ output, const int rows, const int cols)
{
	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 12;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	int vals[12] = { 0 };
	unsigned char frame[3][3];

	if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1)) {
		frame[0][0] = input[(tx - 1) * cols + ty - 1];
		frame[0][1] = input[(tx - 1) * cols + ty];
		frame[0][2] = input[(tx - 1) * cols + ty + 1];
		frame[1][0] = input[tx * cols + ty - 1];
		frame[1][1] = input[tx * cols + ty];
		frame[1][2] = input[tx * cols + ty + 1];
		frame[2][0] = input[(tx + 1) * cols + ty - 1];
		frame[2][1] = input[(tx + 1) * cols + ty];
		frame[2][2] = input[(tx + 1) * cols + ty + 1];

		vals[0] = (CM_Filter[0][0] * frame[0][0]
		+ CM_Filter[0][1] * frame[0][1]
		+ CM_Filter[0][2] * frame[0][2]
		+ CM_Filter[1][0] * frame[1][0]
		+ CM_Filter[1][1] * frame[1][1]
		+ CM_Filter[1][2] * frame[1][2]
		+ CM_Filter[2][0] * frame[2][0]
		+ CM_Filter[2][1] * frame[2][1]
		+ CM_Filter[2][2] * frame[2][2]) >> 4;

		#pragma unroll
		for (int i = 1; i < 12; i++) {
			int _ty = ty + i;
			shift_left(frame);
			if ((tx > 0 && tx < rows - 1) && (_ty > 0 && _ty < cols - 1)) {
				frame[0][2] = input[(tx - 1) * cols + _ty + 1];
				frame[1][2] = input[tx * cols + _ty + 1];
				frame[2][2] = input[(tx + 1) * cols + _ty + 1];

				vals[i] = (CM_Filter[0][0] * frame[0][0]
				+ CM_Filter[0][1] * frame[0][1]
				+ CM_Filter[0][2] * frame[0][2]
				+ CM_Filter[1][0] * frame[1][0]
				+ CM_Filter[1][1] * frame[1][1]
				+ CM_Filter[1][2] * frame[1][2]
				+ CM_Filter[2][0] * frame[2][0]
				+ CM_Filter[2][1] * frame[2][1]
				+ CM_Filter[2][2] * frame[2][2]) >> 4;
			}
		}
		reinterpret_cast<uchar4*>(&output[(tx * cols + ty)])[0] = make_uchar4(vals[0], vals[1], vals[2], vals[3]);
		reinterpret_cast<uchar4*>(&output[(tx * cols + ty + 4)])[0] = make_uchar4(vals[4], vals[5], vals[6], vals[7]);
		reinterpret_cast<uchar4*>(&output[(tx * cols + ty + 8)])[0] = make_uchar4(vals[8], vals[9], vals[10], vals[11]);
	}
}
__global__ void CM_3x3_CF8_Vec(const unsigned char* __restrict__ input, unsigned char* __restrict__ output, const int rows, const int cols)
{
	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	int vals[8] = { 0 };
	unsigned char frame[3][3];

	if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1)) {
		frame[0][0] = input[(tx - 1) * cols + ty - 1];
		frame[0][1] = input[(tx - 1) * cols + ty];
		frame[0][2] = input[(tx - 1) * cols + ty + 1];
		frame[1][0] = input[tx * cols + ty - 1];
		frame[1][1] = input[tx * cols + ty];
		frame[1][2] = input[tx * cols + ty + 1];
		frame[2][0] = input[(tx + 1) * cols + ty - 1];
		frame[2][1] = input[(tx + 1) * cols + ty];
		frame[2][2] = input[(tx + 1) * cols + ty + 1];

		vals[0] = (CM_Filter[0][0] * frame[0][0]
		+ CM_Filter[0][1] * frame[0][1]
		+ CM_Filter[0][2] * frame[0][2]
		+ CM_Filter[1][0] * frame[1][0]
		+ CM_Filter[1][1] * frame[1][1]
		+ CM_Filter[1][2] * frame[1][2]
		+ CM_Filter[2][0] * frame[2][0]
		+ CM_Filter[2][1] * frame[2][1]
		+ CM_Filter[2][2] * frame[2][2]) >> 4;

		#pragma unroll
		for (int i = 1; i < 8; i++) {
			int _ty = ty + i;
			shift_left(frame);
			if ((tx > 0 && tx < rows - 1) && (_ty > 0 && _ty < cols - 1)) {
				frame[0][2] = input[(tx - 1) * cols + _ty + 1];
				frame[1][2] = input[tx * cols + _ty + 1];
				frame[2][2] = input[(tx + 1) * cols + _ty + 1];

				vals[i] = (CM_Filter[0][0] * frame[0][0]
				+ CM_Filter[0][1] * frame[0][1]
				+ CM_Filter[0][2] * frame[0][2]
				+ CM_Filter[1][0] * frame[1][0]
				+ CM_Filter[1][1] * frame[1][1]
				+ CM_Filter[1][2] * frame[1][2]
				+ CM_Filter[2][0] * frame[2][0]
				+ CM_Filter[2][1] * frame[2][1]
				+ CM_Filter[2][2] * frame[2][2]) >> 4;
			}
		}
		reinterpret_cast<uchar4*>(&output[(tx * cols + ty)])[0] = make_uchar4(vals[0], vals[1], vals[2], vals[3]);
		reinterpret_cast<uchar4*>(&output[(tx * cols + ty + 4)])[0] = make_uchar4(vals[4], vals[5], vals[6], vals[7]);
	}
}

__global__ void CM_3x3_CF4_Vec(const unsigned char* __restrict__ input, unsigned char* __restrict__ output, const int rows, const int cols)
{
	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	int vals[4] = { 0 };
	unsigned char frame[3][3];

	if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1)) {
		frame[0][0] = input[(tx - 1) * cols + ty - 1];
		frame[0][1] = input[(tx - 1) * cols + ty];
		frame[0][2] = input[(tx - 1) * cols + ty + 1];
		frame[1][0] = input[tx * cols + ty - 1];
		frame[1][1] = input[tx * cols + ty];
		frame[1][2] = input[tx * cols + ty + 1];
		frame[2][0] = input[(tx + 1) * cols + ty - 1];
		frame[2][1] = input[(tx + 1) * cols + ty];
		frame[2][2] = input[(tx + 1) * cols + ty + 1];

		vals[0] = (CM_Filter[0][0] * frame[0][0]
		+ CM_Filter[0][1] * frame[0][1]
		+ CM_Filter[0][2] * frame[0][2]
		+ CM_Filter[1][0] * frame[1][0]
		+ CM_Filter[1][1] * frame[1][1]
		+ CM_Filter[1][2] * frame[1][2]
		+ CM_Filter[2][0] * frame[2][0]
		+ CM_Filter[2][1] * frame[2][1]
		+ CM_Filter[2][2] * frame[2][2]) >> 4;

		#pragma unroll
		for (int i = 1; i < 4; i++) {
			int _ty = ty + i;
			shift_left(frame);
			if ((tx > 0 && tx < rows - 1) && (_ty > 0 && _ty < cols - 1)) {
				frame[0][2] = input[(tx - 1) * cols + _ty + 1];
				frame[1][2] = input[tx * cols + _ty + 1];
				frame[2][2] = input[(tx + 1) * cols + _ty + 1];

				vals[i] = (CM_Filter[0][0] * frame[0][0]
				+ CM_Filter[0][1] * frame[0][1]
				+ CM_Filter[0][2] * frame[0][2]
				+ CM_Filter[1][0] * frame[1][0]
				+ CM_Filter[1][1] * frame[1][1]
				+ CM_Filter[1][2] * frame[1][2]
				+ CM_Filter[2][0] * frame[2][0]
				+ CM_Filter[2][1] * frame[2][1]
				+ CM_Filter[2][2] * frame[2][2]) >> 4;
			}
		}
		reinterpret_cast<uchar4*>(&output[(tx * cols + ty)])[0] = make_uchar4(vals[0], vals[1], vals[2], vals[3]);
	}
}

__global__ void CM_3x3_CF2_Vec(const unsigned char* __restrict__ input, unsigned char* __restrict__ output, const int rows, const int cols)
{
	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	int vals[2] = { 0 };
	unsigned char frame[3][3];

	if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1)) {
		frame[0][0] = input[(tx - 1) * cols + ty - 1];
		frame[0][1] = input[(tx - 1) * cols + ty];
		frame[0][2] = input[(tx - 1) * cols + ty + 1];
		frame[1][0] = input[tx * cols + ty - 1];
		frame[1][1] = input[tx * cols + ty];
		frame[1][2] = input[tx * cols + ty + 1];
		frame[2][0] = input[(tx + 1) * cols + ty - 1];
		frame[2][1] = input[(tx + 1) * cols + ty];
		frame[2][2] = input[(tx + 1) * cols + ty + 1];

		vals[0] = (CM_Filter[0][0] * frame[0][0]
		+ CM_Filter[0][1] * frame[0][1]
		+ CM_Filter[0][2] * frame[0][2]
		+ CM_Filter[1][0] * frame[1][0]
		+ CM_Filter[1][1] * frame[1][1]
		+ CM_Filter[1][2] * frame[1][2]
		+ CM_Filter[2][0] * frame[2][0]
		+ CM_Filter[2][1] * frame[2][1]
		+ CM_Filter[2][2] * frame[2][2]) >> 4;

		#pragma unroll
		for (int i = 1; i < 2; i++) {
			int _ty = ty + i;
			shift_left(frame);
			if ((tx > 0 && tx < rows - 1) && (_ty > 0 && _ty < cols - 1)) {
				frame[0][2] = input[(tx - 1) * cols + _ty + 1];
				frame[1][2] = input[tx * cols + _ty + 1];
				frame[2][2] = input[(tx + 1) * cols + _ty + 1];

				vals[i] = (CM_Filter[0][0] * frame[0][0]
				+ CM_Filter[0][1] * frame[0][1]
				+ CM_Filter[0][2] * frame[0][2]
				+ CM_Filter[1][0] * frame[1][0]
				+ CM_Filter[1][1] * frame[1][1]
				+ CM_Filter[1][2] * frame[1][2]
				+ CM_Filter[2][0] * frame[2][0]
				+ CM_Filter[2][1] * frame[2][1]
				+ CM_Filter[2][2] * frame[2][2]) >> 4;
			}
		}
		reinterpret_cast<uchar2*>(&output[(tx * cols + ty)])[0] = make_uchar2(vals[0], vals[1]);
	}
}

__global__ void SM_3x3_CF16_Vec(unsigned char* __restrict__ input, unsigned char* __restrict__ output, const int rows, const int cols)
{
	__shared__  unsigned char cache[34][514];

	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 16;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	int cy = threadIdx.x * 16 + 1;
	int cx = threadIdx.y + 1;

	uchar4 u4;

	if ((tx > 0 && tx < rows - 1) && (ty < cols - 1)) {
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
		unsigned char frame[3][3];

		frame[0][0] = cache[cx - 1][cy - 1];
		frame[0][1] = cache[cx - 1][cy];
		frame[0][2] = cache[cx - 1][cy + 1];
		frame[1][0] = cache[cx][cy - 1];
		frame[1][1] = cache[cx][cy];
		frame[1][2] = cache[cx][cy + 1];
		frame[2][0] = cache[cx + 1][cy - 1];
		frame[2][1] = cache[cx + 1][cy];
		frame[2][2] = cache[cx + 1][cy + 1];

		vals[0] = (GM_Filter[0][0] * frame[0][0]
		+ GM_Filter[0][1] * frame[0][1]
		+ GM_Filter[0][2] * frame[0][2]
		+ GM_Filter[1][0] * frame[1][0]
		+ GM_Filter[1][1] * frame[1][1]
		+ GM_Filter[1][2] * frame[1][2]
		+ GM_Filter[2][0] * frame[2][0]
		+ GM_Filter[2][1] * frame[2][1]
		+ GM_Filter[2][2] * frame[2][2]) >> 4;

		#pragma unroll
		for (int i = 1; i < 16; i++) {
			int _ty = ty + i;
			int _cy = cy + i;
			shift_left(frame);
			frame[0][2] = cache[cx - 1][_cy + 1];
			frame[1][2] = cache[cx][_cy + 1];
			frame[2][2] = cache[cx + 1][_cy + 1];

			if (_ty < cols - 1) {
				vals[i] = (GM_Filter[0][0] * frame[0][0]
				+ GM_Filter[0][1] * frame[0][1]
				+ GM_Filter[0][2] * frame[0][2]
				+ GM_Filter[1][0] * frame[1][0]
				+ GM_Filter[1][1] * frame[1][1]
				+ GM_Filter[1][2] * frame[1][2]
				+ GM_Filter[2][0] * frame[2][0]
				+ GM_Filter[2][1] * frame[2][1]
				+ GM_Filter[2][2] * frame[2][2]) >> 4;
			}
		}
		reinterpret_cast<uchar4*>(&output[(tx * cols + ty)])[0] = make_uchar4(vals[0], vals[1], vals[2], vals[3]);
		reinterpret_cast<uchar4*>(&output[(tx * cols + ty + 4)])[0] = make_uchar4(vals[4], vals[5], vals[6], vals[7]);
		reinterpret_cast<uchar4*>(&output[(tx * cols + ty + 8)])[0] = make_uchar4(vals[8], vals[9], vals[10], vals[11]);
		reinterpret_cast<uchar4*>(&output[(tx * cols + ty + 12)])[0] = make_uchar4(vals[12], vals[13], vals[14], vals[15]);
	}
}

__global__ void SM_3x3_CF12_Vec(unsigned char* __restrict__ input, unsigned char* __restrict__ output, const int rows, const int cols)
{
	__shared__  unsigned char cache[34][386];

	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 12;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	int cy = threadIdx.x * 12 + 1;
	int cx = threadIdx.y + 1;
	
	uchar4 u4;

	if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1)) {
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
		unsigned char frame[3][3];

		frame[0][0] = cache[cx - 1][cy - 1];
		frame[0][1] = cache[cx - 1][cy];
		frame[0][2] = cache[cx - 1][cy + 1];
		frame[1][0] = cache[cx][cy - 1];
		frame[1][1] = cache[cx][cy];
		frame[1][2] = cache[cx][cy + 1];
		frame[2][0] = cache[cx + 1][cy - 1];
		frame[2][1] = cache[cx + 1][cy];
		frame[2][2] = cache[cx + 1][cy + 1];

		vals[0] = (GM_Filter[0][0] * frame[0][0]
		+ GM_Filter[0][1] * frame[0][1]
		+ GM_Filter[0][2] * frame[0][2]
		+ GM_Filter[1][0] * frame[1][0]
		+ GM_Filter[1][1] * frame[1][1]
		+ GM_Filter[1][2] * frame[1][2]
		+ GM_Filter[2][0] * frame[2][0]
		+ GM_Filter[2][1] * frame[2][1]
		+ GM_Filter[2][2] * frame[2][2] >> 4);

		#pragma unroll
		for (int i = 1; i < 12; i++) {
			int _ty = ty + i;
			int _cy = cy + i;
			shift_left(frame);
			frame[0][2] = cache[cx - 1][_cy + 1];
			frame[1][2] = cache[cx][_cy + 1];
			frame[2][2] = cache[cx + 1][_cy + 1];

			if (_ty < cols - 1) {
				vals[i] = (GM_Filter[0][0] * frame[0][0]
				+ GM_Filter[0][1] * frame[0][1]
				+ GM_Filter[0][2] * frame[0][2]
				+ GM_Filter[1][0] * frame[1][0]
				+ GM_Filter[1][1] * frame[1][1]
				+ GM_Filter[1][2] * frame[1][2]
				+ GM_Filter[2][0] * frame[2][0]
				+ GM_Filter[2][1] * frame[2][1]
				+ GM_Filter[2][2] * frame[2][2]) >> 4;
			}
		}
		reinterpret_cast<uchar4*>(&output[(tx * cols + ty)])[0] = make_uchar4(vals[0], vals[1], vals[2], vals[3]);
		reinterpret_cast<uchar4*>(&output[(tx * cols + ty + 4)])[0] = make_uchar4(vals[4], vals[5], vals[6], vals[7]);
		reinterpret_cast<uchar4*>(&output[(tx * cols + ty + 8)])[0] = make_uchar4(vals[8], vals[9], vals[10], vals[11]);
	}
}

__global__ void SM_3x3_CF8_Vec(unsigned char* __restrict__ input, unsigned char* __restrict__ output, const int rows, const int cols)
{
	__shared__  unsigned char cache[34][258];

	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	int cy = threadIdx.x * 8 + 1;
	int cx = threadIdx.y + 1;

	uchar4 u4;

	if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1)) {
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
		unsigned char frame[3][3];

		frame[0][0] = cache[cx - 1][cy - 1];
		frame[0][1] = cache[cx - 1][cy];
		frame[0][2] = cache[cx - 1][cy + 1];
		frame[1][0] = cache[cx][cy - 1];
		frame[1][1] = cache[cx][cy];
		frame[1][2] = cache[cx][cy + 1];
		frame[2][0] = cache[cx + 1][cy - 1];
		frame[2][1] = cache[cx + 1][cy];
		frame[2][2] = cache[cx + 1][cy + 1];

		vals[0] = (GM_Filter[0][0] * frame[0][0]
		+ GM_Filter[0][1] * frame[0][1]
		+ GM_Filter[0][2] * frame[0][2]
		+ GM_Filter[1][0] * frame[1][0]
		+ GM_Filter[1][1] * frame[1][1]
		+ GM_Filter[1][2] * frame[1][2]
		+ GM_Filter[2][0] * frame[2][0]
		+ GM_Filter[2][1] * frame[2][1]
		+ GM_Filter[2][2] * frame[2][2]) >> 4;

		#pragma unroll
		for (int i = 1; i < 8; i++) {
			int _ty = ty + i;
			int _cy = cy + i;
			shift_left(frame);
			frame[0][2] = cache[cx - 1][_cy + 1];
			frame[1][2] = cache[cx][_cy + 1];
			frame[2][2] = cache[cx + 1][_cy + 1];

			if (_ty < cols - 1) {
				vals[i] = (GM_Filter[0][0] * frame[0][0]
				+ GM_Filter[0][1] * frame[0][1]
				+ GM_Filter[0][2] * frame[0][2]
				+ GM_Filter[1][0] * frame[1][0]
				+ GM_Filter[1][1] * frame[1][1]
				+ GM_Filter[1][2] * frame[1][2]
				+ GM_Filter[2][0] * frame[2][0]
				+ GM_Filter[2][1] * frame[2][1]
				+ GM_Filter[2][2] * frame[2][2]) >> 4;
			}
		}
		reinterpret_cast<uchar4*>(&output[(tx * cols + ty)])[0] = make_uchar4(vals[0], vals[1], vals[2], vals[3]);
		reinterpret_cast<uchar4*>(&output[(tx * cols + ty + 4)])[0] = make_uchar4(vals[4], vals[5], vals[6], vals[7]);
	}
}

__global__ void SM_3x3_CF4_Vec(unsigned char* __restrict__ input, unsigned char* __restrict__ output, const int rows, const int cols)
{
	__shared__  unsigned char cache[34][130];

	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	int cy = threadIdx.x * 4 + 1;
	int cx = threadIdx.y + 1;

	uchar4 u4;

	if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1)) {
		u4 = reinterpret_cast<uchar4*>(&input[tx * cols + ty])[0];
		cache[cx][cy] = u4.x;
		cache[cx][cy + 1] = u4.y;
		cache[cx][cy + 2] = u4.z;
		cache[cx][cy + 3] = u4.w;
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
		unsigned char frame[3][3];

		frame[0][0] = cache[cx - 1][cy - 1];
		frame[0][1] = cache[cx - 1][cy];
		frame[0][2] = cache[cx - 1][cy + 1];
		frame[1][0] = cache[cx][cy - 1];
		frame[1][1] = cache[cx][cy];
		frame[1][2] = cache[cx][cy + 1];
		frame[2][0] = cache[cx + 1][cy - 1];
		frame[2][1] = cache[cx + 1][cy];
		frame[2][2] = cache[cx + 1][cy + 1];

		vals[0] = (GM_Filter[0][0] * frame[0][0]
		+ GM_Filter[0][1] * frame[0][1]
		+ GM_Filter[0][2] * frame[0][2]
		+ GM_Filter[1][0] * frame[1][0]
		+ GM_Filter[1][1] * frame[1][1]
		+ GM_Filter[1][2] * frame[1][2]
		+ GM_Filter[2][0] * frame[2][0]
		+ GM_Filter[2][1] * frame[2][1]
		+ GM_Filter[2][2] * frame[2][2]) >> 4;

		#pragma unroll
		for (int i = 1; i < 4; i++) {
			int _ty = ty + i;
			int _cy = cy + i;
			shift_left(frame);
			frame[0][2] = cache[cx - 1][_cy + 1];
			frame[1][2] = cache[cx][_cy + 1];
			frame[2][2] = cache[cx + 1][_cy + 1];

			if (_ty < cols - 1) {
				vals[i] = (GM_Filter[0][0] * frame[0][0]
				+ GM_Filter[0][1] * frame[0][1]
				+ GM_Filter[0][2] * frame[0][2]
				+ GM_Filter[1][0] * frame[1][0]
				+ GM_Filter[1][1] * frame[1][1]
				+ GM_Filter[1][2] * frame[1][2]
				+ GM_Filter[2][0] * frame[2][0]
				+ GM_Filter[2][1] * frame[2][1]
				+ GM_Filter[2][2] * frame[2][2]) >> 4;
			}
		}
		reinterpret_cast<uchar4*>(&output[(tx * cols + ty)])[0] = make_uchar4(vals[0], vals[1], vals[2], vals[3]);
	}
}


__global__ void SM_3x3_CF2_Vec(unsigned char* __restrict__ input, unsigned char* __restrict__ output, const int rows, const int cols)
{
	__shared__  unsigned char cache[34][66];

	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	int cy = threadIdx.x * 2 + 1;
	int cx = threadIdx.y + 1;

	uchar2 u2;

	if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1)) {
		u2 = reinterpret_cast<uchar2*>(&input[tx * cols + ty])[0];
		
		cache[cx][cy] = u2.x;
		cache[cx][cy + 1] = u2.y;
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
		unsigned char frame[3][3];

		frame[0][0] = cache[cx - 1][cy - 1];
		frame[0][1] = cache[cx - 1][cy];
		frame[0][2] = cache[cx - 1][cy + 1];
		frame[1][0] = cache[cx][cy - 1];
		frame[1][1] = cache[cx][cy];
		frame[1][2] = cache[cx][cy + 1];
		frame[2][0] = cache[cx + 1][cy - 1];
		frame[2][1] = cache[cx + 1][cy];
		frame[2][2] = cache[cx + 1][cy + 1];

		vals[0] = (GM_Filter[0][0] * frame[0][0]
		+ GM_Filter[0][1] * frame[0][1]
		+ GM_Filter[0][2] * frame[0][2]
		+ GM_Filter[1][0] * frame[1][0]
		+ GM_Filter[1][1] * frame[1][1]
		+ GM_Filter[1][2] * frame[1][2]
		+ GM_Filter[2][0] * frame[2][0]
		+ GM_Filter[2][1] * frame[2][1]
		+ GM_Filter[2][2] * frame[2][2]) >> 4;

		#pragma unroll
		for (int i = 1; i < 2; i++) {
			int _ty = ty + i;
			int _cy = cy + i;
			shift_left(frame);
			frame[0][2] = cache[cx - 1][_cy + 1];
			frame[1][2] = cache[cx][_cy + 1];
			frame[2][2] = cache[cx + 1][_cy + 1];

			if (_ty < cols - 1) {
				vals[i] = (GM_Filter[0][0] * frame[0][0]
					+ GM_Filter[0][1] * frame[0][1]
					+ GM_Filter[0][2] * frame[0][2]
					+ GM_Filter[1][0] * frame[1][0]
					+ GM_Filter[1][1] * frame[1][1]
					+ GM_Filter[1][2] * frame[1][2]
					+ GM_Filter[2][0] * frame[2][0]
					+ GM_Filter[2][1] * frame[2][1]
					+ GM_Filter[2][2] * frame[2][2]) >> 4;
			}
		}
		reinterpret_cast<uchar2*>(&output[(tx * cols + ty)])[0] = make_uchar2(vals[0], vals[1]);
	}
}

void launch_kernels(cv::Mat* input_img, cv::Mat* output_img)
{
	unsigned char* d_input = nullptr;
	unsigned char* d_output = nullptr;
	unsigned char* h_input = input_img->data;
	unsigned char* h_output = output_img->data;

	const int cols = (*input_img).cols;
	const int rows = (*input_img).rows;
	const int size = cols * rows * sizeof(unsigned char);

	dim3 block(32,32);
	dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
	dim3 grid_cf2(((cols / 2) + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
	dim3 grid_cf4(((cols / 4) + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
	dim3 grid_cf8(((cols / 8) + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
	dim3 grid_cf12(((cols / 12) + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
	dim3 grid_cf16(((cols / 16) + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

	CHECK_CUDA_ERROR(cudaHostRegister(h_output, size, cudaHostRegisterPortable));
	CHECK_CUDA_ERROR(cudaHostRegister(h_input, size, cudaHostRegisterPortable));
	CHECK_CUDA_ERROR(cudaMalloc((void**)&d_input, size));
	CHECK_CUDA_ERROR(cudaMalloc((void**)&d_output, size));
	CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));
	
	#define BASE_KERNELS 0
	#define COARSENED_KERNELS 0
	#define VECTORIZED_KERNELS 0
	#define NPP_KERNEL 1

	#if BASE_KERNELS
	GM_3x3 << <grid, block >> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	cv::imwrite("../images/outputs/GM_3x3.png", *output_img);
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	CM_3x3 << <grid, block >> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	cv::imwrite("../images/outputs/CM_3x3.png", *output_img);
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	SM_3x3 << <grid, block >> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	cv::imwrite("../images/outputs/SM_3x3.png", *output_img);
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));
	#endif

	#if COARSENED_KERNELS
	GM_3x3_CF2 << <grid_cf2, block >> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	cv::imwrite("../images/outputs/GM_3x3_CF2.png", *output_img);
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	GM_3x3_CF4 << <grid_cf4, block >> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	cv::imwrite("../images/outputs/GM_3x3_CF4.png", *output_img);
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	GM_3x3_CF8 << <grid_cf8, block >> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	cv::imwrite("../images/outputs/GM_3x3_CF8.png", *output_img);
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	GM_3x3_CF12 << <grid_cf12, block >> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	cv::imwrite("../images/outputs/GM_3x3_CF12.png", *output_img);
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	GM_3x3_CF16 << <grid_cf16, block >> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	cv::imwrite("../images/outputs/GM_3x3_CF16.png", *output_img);

	CM_3x3_CF2 << <grid_cf2, block >> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	cv::imwrite("../images/outputs/CM_3x3_CF2.png", *output_img);
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	CM_3x3_CF4 << <grid_cf4, block >> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	cv::imwrite("../images/outputs/CM_3x3_CF4.png", *output_img);
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	CM_3x3_CF8 << <grid_cf8, block >> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	cv::imwrite("../images/outputs/CM_3x3_CF8.png", *output_img);
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	CM_3x3_CF12 << <grid_cf12, block >> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	cv::imwrite("../images/outputs/CM_3x3_CF12.png", *output_img);
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	CM_3x3_CF16 << <grid_cf16, block >> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	cv::imwrite("../images/outputs/CM_3x3_CF16.png", *output_img);

	SM_3x3_CF2 << <grid_cf2, block >> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	cv::imwrite("../images/outputs/SM_3x3_CF2.png", *output_img);
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	SM_3x3_CF4 << <grid_cf4, block >> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	cv::imwrite("../images/outputs/SM_3x3_CF4.png", *output_img);
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	SM_3x3_CF8 << <grid_cf8, block >> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	cv::imwrite("../images/outputs/SM_3x3_CF8.png", *output_img);
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	SM_3x3_CF12 << <grid_cf12, block >> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	cv::imwrite("../images/outputs/SM_3x3_CF12.png", *output_img);
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	SM_3x3_CF16 << <grid_cf16, block >> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	cv::imwrite("../images/outputs/SM_3x3_CF16.png", *output_img);
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));
	#endif

	#if VECTORIZED_KERNELS
	GM_3x3_CF2_Vec << <grid_cf2, block>> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	cv::imwrite("../images/outputs/GM_3x3_CF2_V.png", *output_img);
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	GM_3x3_CF4_Vec << <grid_cf4, block>> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	cv::imwrite("../images/outputs/GM_3x3_CF4_V.png", *output_img);
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	GM_3x3_CF8_Vec << <grid_cf8, block>> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	cv::imwrite("../images/outputs/GM_3x3_CF8_V.png", *output_img);
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	GM_3x3_CF12_Vec << <grid_cf12, block>> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	cv::imwrite("../images/outputs/GM_3x3_CF12_V.png", *output_img);
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	GM_3x3_CF16_Vec << <grid_cf16, block>> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	cv::imwrite("../images/outputs/GM_3x3_CF16_V.png", *output_img);
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	CM_3x3_CF2_Vec << <grid_cf2, block>> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	cv::imwrite("../images/outputs/CM_3x3_CF2_V.png", *output_img);
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	CM_3x3_CF4_Vec << <grid_cf4, block>> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	cv::imwrite("../images/outputs/CM_3x3_CF4_V.png", *output_img);
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	CM_3x3_CF8_Vec << <grid_cf8, block>> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	cv::imwrite("../images/outputs/CM_3x3_CF8_V.png", *output_img);
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	CM_3x3_CF12_Vec << <grid_cf12, block>> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	cv::imwrite("../images/outputs/CM_3x3_CF12_V.png", *output_img);
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	CM_3x3_CF16_Vec << <grid_cf16, block>> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	cv::imwrite("../images/outputs/CM_3x3_CF16_V.png", *output_img);
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	SM_3x3_CF2_Vec << <grid_cf2, block>> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	cv::imwrite("../images/outputs/SM_3x3_CF2_V.png", *output_img);
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	SM_3x3_CF4_Vec << <grid_cf4, block>> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	cv::imwrite("../images/outputs/SM_3x3_CF4_V.png", *output_img);
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	SM_3x3_CF8_Vec << <grid_cf8, block>> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	cv::imwrite("../images/outputs/SM_3x3_CF8_V.png", *output_img);
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	SM_3x3_CF12_Vec << <grid_cf12, block>> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	cv::imwrite("../images/outputs/SM_3x3_CF12_V.png", *output_img);
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	SM_3x3_CF16_Vec << <grid_cf16, block>> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	cv::imwrite("../images/outputs/SM_3x3_CF16_V.png", *output_img);
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));
	#endif

	#if NPP_KERNEL
	Npp32s step = cols * rows;
	const Npp32s h_kernel[9] = {1,2,1,2,4,2,1,2,1};
	Npp32s* d_kernel = NULL;

	nppiFilterGauss_8u_C1R(d_input, input_img->step, d_output, input_img->step, {cols, rows}, NPP_MASK_SIZE_3_X_3);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	cv::imwrite("../images/outputs/nppiFilterGauss_8u_C1R.png", *output_img);
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	CHECK_CUDA_ERROR(cudaMalloc(&d_kernel, sizeof(Npp32s) * 9));
	CHECK_CUDA_ERROR(cudaMemcpy(d_kernel, h_kernel, sizeof(Npp32s) * 9, cudaMemcpyHostToDevice));
	nppiFilter_8u_C1R(d_input, cols * sizeof(unsigned char), d_output, cols * sizeof(unsigned char), {cols, rows}, d_kernel, {3,3}, {1,1}, 16);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	cv::imwrite("../images/outputs/nppiFilter_8u_C1R.png", *output_img);
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));
	#endif

	#if CUDNN_KERNEL

	#endif

	cudaHostUnregister(h_input);
	cudaHostUnregister(h_output);
	cudaFree(d_input);
	cudaFree(d_output);
	cudaDeviceReset();
}