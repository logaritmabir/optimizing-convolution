#include "kernels.cuh"

#define BLOCKDIMX 8
#define BLOCKDIMY 16

inline __device__ void shift_left(imtype arr[][3]) {
	arr[0][0] = arr[0][1];
	arr[1][0] = arr[1][1];
	arr[2][0] = arr[2][1];
	arr[0][1] = arr[0][2];
	arr[1][1] = arr[1][2];
	arr[2][1] = arr[2][2];
}

__constant__ float CM_Filter[3][3] = {{1 / 16.0f, 2 / 16.0f, 1 / 16.0f},
									   {2 / 16.0f, 4 / 16.0f, 2 / 16.0f},
									   {1 / 16.0f, 2 / 16.0f, 1 / 16.0f}};
__device__ float GM_Filter[3][3] = {{1 / 16.0f, 2 / 16.0f, 1 / 16.0f},
									 {2 / 16.0f, 4 / 16.0f, 2 / 16.0f},
									 {1 / 16.0f, 2 / 16.0f, 1 / 16.0f}};

__global__ void GM_3x3(const imtype* __restrict__ input, imtype* __restrict__ output, const int rows, const int cols){
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
		+ GM_Filter[2][2] * input[(tx + 1) * cols + ty + 1]);
	}
}

__global__ void CM_3x3(const imtype* __restrict__ input, imtype* __restrict__ output, const int rows, const int cols)
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
		+ CM_Filter[2][2] * input[(tx + 1) * cols + ty + 1]);
	}
}

__global__ void SM_3x3(const imtype* __restrict__ input, imtype* __restrict__ output, const int rows, const int cols)
{
	__shared__  imtype cache[BLOCKDIMY + 2][BLOCKDIMX + 2];

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
    if (cx == BLOCKDIMY && tx < rows - 1) {
        cache[BLOCKDIMY + 1][cy] = input[(tx + 1) * cols + ty];
    }
    if (cy == 1 && ty > 0) {
        cache[cx][0] = input[tx * cols + ty - 1];
    }
    if (cy == BLOCKDIMX && ty < cols - 1) {
        cache[cx][BLOCKDIMX + 1] = input[tx * cols + ty + 1];
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
		+ GM_Filter[2][2] * cache[cx + 1][cy + 1]);
	}
}

__global__ void GM_3x3_CF16(const imtype* __restrict__ input, imtype* __restrict__ output, const int rows, const int cols)
{
	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 16;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	imtype frame[3][3];

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
		+ GM_Filter[2][2] * frame[2][2]);

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
				+ GM_Filter[2][2] * frame[2][2]);
			}
		}
	}
}

__global__ void GM_3x3_CF12(const imtype* __restrict__ input, imtype* __restrict__ output, const int rows, const int cols)
{
	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 12;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	imtype frame[3][3];

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
		+ GM_Filter[2][2] * frame[2][2]);

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
				+ GM_Filter[2][2] * frame[2][2]);
			}
		}
	}
}

__global__ void GM_3x3_CF8(const imtype* __restrict__ input, imtype* __restrict__ output, const int rows, const int cols)
{
	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	imtype frame[3][3];

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
		+ GM_Filter[2][2] * frame[2][2]);

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
				+ GM_Filter[2][2] * frame[2][2]);
			}
		}
	}
}

__global__ void GM_3x3_CF4(const imtype* __restrict__ input, imtype* __restrict__ output, const int rows, const int cols)
{
	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;
	imtype frame[3][3];

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
			+ GM_Filter[2][2] * frame[2][2]);

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
					+ GM_Filter[2][2] * frame[2][2]);
			}
		}
	}
}

__global__ void GM_3x3_CF2(const imtype* __restrict__ input, imtype* __restrict__ output, const int rows, const int cols)
{
	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	imtype frame[3][3];

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
		+ GM_Filter[2][2] * frame[2][2]);

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
					+ GM_Filter[2][2] * frame[2][2]);
			}
		}
	}
}

__global__ void CM_3x3_CF16(const imtype* __restrict__ input, imtype* __restrict__ output, const int rows, const int cols)
{
	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 16;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	imtype frame[3][3];

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
			+ CM_Filter[2][2] * frame[2][2]);

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
					+ CM_Filter[2][2] * frame[2][2]);
			}
		}
	}
}

__global__ void CM_3x3_CF12(const imtype* __restrict__ input, imtype* __restrict__ output, const int rows, const int cols)
{
	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 12;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;
	imtype frame[3][3];

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
			+ CM_Filter[2][2] * frame[2][2]);

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
					+ CM_Filter[2][2] * frame[2][2]);
			}
		}
	}
}

__global__ void CM_3x3_CF8(const imtype* __restrict__ input, imtype* __restrict__ output, const int rows, const int cols)
{
	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;
	imtype frame[3][3];

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
			+ CM_Filter[2][2] * frame[2][2]);

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
					+ CM_Filter[2][2] * frame[2][2]);
			}
		}
	}
}

__global__ void CM_3x3_CF4(const imtype* __restrict__ input, imtype* __restrict__ output, const int rows, const int cols)
{
	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	imtype frame[3][3];

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
			+ CM_Filter[2][2] * frame[2][2]);

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
					+ CM_Filter[2][2] * frame[2][2]);
			}
		}
	}
}

__global__ void CM_3x3_CF2(const imtype* __restrict__ input, imtype* __restrict__ output, const int rows, const int cols)
{
	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	imtype frame[3][3];

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
			+ CM_Filter[2][2] * frame[2][2]);

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
					+ CM_Filter[2][2] * frame[2][2]);
			}
		}
	}
}

__global__ void GM_3x3_CF16_Vec(const imtype* __restrict__ input, imtype* __restrict__ output, const int rows, const int cols)
{
	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 16;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	imtype vals[16] = { 0 };
	imtype frame[3][3];

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
			+ GM_Filter[2][2] * frame[2][2]);

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
					+ GM_Filter[2][2] * frame[2][2]);
			}
		}
		reinterpret_cast<imtype4*>(&output[(tx * cols + ty)])[0] = make_imtype4(vals[0], vals[1], vals[2], vals[3]);
		reinterpret_cast<imtype4*>(&output[(tx * cols + ty + 4)])[0] = make_imtype4(vals[4], vals[5], vals[6], vals[7]);
		reinterpret_cast<imtype4*>(&output[(tx * cols + ty + 8)])[0] = make_imtype4(vals[8], vals[9], vals[10], vals[11]);
		reinterpret_cast<imtype4*>(&output[(tx * cols + ty + 12)])[0] = make_imtype4(vals[12], vals[13], vals[14], vals[15]);
	}
}

__global__ void GM_3x3_CF12_Vec(const imtype* __restrict__ input, imtype* __restrict__ output, const int rows, const int cols)
{
	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 12;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	imtype vals[12] = { 0 };
	imtype frame[3][3];

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
			+ GM_Filter[2][2] * frame[2][2]);

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
					+ GM_Filter[2][2] * frame[2][2]);
			}
		}
		reinterpret_cast<imtype4*>(&output[(tx * cols + ty)])[0] = make_imtype4(vals[0], vals[1], vals[2], vals[3]);
		reinterpret_cast<imtype4*>(&output[(tx * cols + ty + 4)])[0] = make_imtype4(vals[4], vals[5], vals[6], vals[7]);
		reinterpret_cast<imtype4*>(&output[(tx * cols + ty + 8)])[0] = make_imtype4(vals[8], vals[9], vals[10], vals[11]);
	}
}
__global__ void GM_3x3_CF8_Vec(const imtype* __restrict__ input, imtype* __restrict__ output, const int rows, const int cols)
{
	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	imtype vals[8] = { 0 };
	imtype frame[3][3];

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
			+ GM_Filter[2][2] * frame[2][2]);

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
					+ GM_Filter[2][2] * frame[2][2]);
			}
		}
		reinterpret_cast<imtype4*>(&output[(tx * cols + ty)])[0] = make_imtype4(vals[0], vals[1], vals[2], vals[3]);
		reinterpret_cast<imtype4*>(&output[(tx * cols + ty + 4)])[0] = make_imtype4(vals[4], vals[5], vals[6], vals[7]);
	}
}

__global__ void GM_3x3_CF4_Vec(const imtype* __restrict__ input, imtype* __restrict__ output, const int rows, const int cols)
{
	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	imtype vals[4] = { 0 };

	imtype frame[3][3];

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
			+ GM_Filter[2][2] * frame[2][2]); 

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
					+ GM_Filter[2][2] * frame[2][2]);
			}
		}
		reinterpret_cast<imtype4*>(&output[(tx * cols + ty)])[0] = make_imtype4(vals[0], vals[1], vals[2], vals[3]);
	}
}

__global__ void GM_3x3_CF2_Vec(const imtype* __restrict__ input, imtype* __restrict__ output, const int rows, const int cols)
{
	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	imtype vals[2] = { 0 };

	imtype frame[3][3];

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
			+ GM_Filter[2][2] * frame[2][2]);

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
					+ GM_Filter[2][2] * frame[2][2]);
			}
		}
		reinterpret_cast<imtype2*>(&output[(tx * cols + ty)])[0] = make_imtype2(vals[0], vals[1]);
	}
}

__global__ void CM_3x3_CF16_Vec(const imtype* __restrict__ input, imtype* __restrict__ output, const int rows, const int cols)
{
	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 16;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	float vals[16] = { 0 };
	imtype frame[3][3];

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
			+ CM_Filter[2][2] * frame[2][2]);

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
					+ CM_Filter[2][2] * frame[2][2]);
			}
		}
		reinterpret_cast<imtype4*>(&output[(tx * cols + ty)])[0] = make_imtype4(vals[0], vals[1], vals[2], vals[3]);
		reinterpret_cast<imtype4*>(&output[(tx * cols + ty + 4)])[0] = make_imtype4(vals[4], vals[5], vals[6], vals[7]);
		reinterpret_cast<imtype4*>(&output[(tx * cols + ty + 8)])[0] = make_imtype4(vals[8], vals[9], vals[10], vals[11]);
		reinterpret_cast<imtype4*>(&output[(tx * cols + ty + 12)])[0] = make_imtype4(vals[12], vals[13], vals[14], vals[15]);
	}
}

__global__ void CM_3x3_CF12_Vec(const imtype* __restrict__ input, imtype* __restrict__ output, const int rows, const int cols)
{
	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 12;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	imtype vals[12] = { 0 };
	imtype frame[3][3];

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
		+ CM_Filter[2][2] * frame[2][2]);

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
				+ CM_Filter[2][2] * frame[2][2]);
			}
		}
		reinterpret_cast<imtype4*>(&output[(tx * cols + ty)])[0] = make_imtype4(vals[0], vals[1], vals[2], vals[3]);
		reinterpret_cast<imtype4*>(&output[(tx * cols + ty + 4)])[0] = make_imtype4(vals[4], vals[5], vals[6], vals[7]);
		reinterpret_cast<imtype4*>(&output[(tx * cols + ty + 8)])[0] = make_imtype4(vals[8], vals[9], vals[10], vals[11]);
	}
}
__global__ void CM_3x3_CF8_Vec(const imtype* __restrict__ input, imtype* __restrict__ output, const int rows, const int cols)
{
	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	imtype vals[8] = { 0 };
	imtype frame[3][3];

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
		+ CM_Filter[2][2] * frame[2][2]);

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
				+ CM_Filter[2][2] * frame[2][2]);
			}
		}
		reinterpret_cast<imtype4*>(&output[(tx * cols + ty)])[0] = make_imtype4(vals[0], vals[1], vals[2], vals[3]);
		reinterpret_cast<imtype4*>(&output[(tx * cols + ty + 4)])[0] = make_imtype4(vals[4], vals[5], vals[6], vals[7]);
	}
}

__global__ void CM_3x3_CF4_Vec(const imtype* __restrict__ input, imtype* __restrict__ output, const int rows, const int cols)
{
	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	imtype vals[4] = { 0 };
	imtype frame[3][3];

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
		+ CM_Filter[2][2] * frame[2][2]);

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
				+ CM_Filter[2][2] * frame[2][2]);
			}
		}
		reinterpret_cast<imtype4*>(&output[(tx * cols + ty)])[0] = make_imtype4(vals[0], vals[1], vals[2], vals[3]);
	}
}

__global__ void CM_3x3_CF2_Vec(const imtype* __restrict__ input, imtype* __restrict__ output, const int rows, const int cols)
{
	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	imtype vals[2] = { 0 };
	imtype frame[3][3];

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
		+ CM_Filter[2][2] * frame[2][2]);

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
				+ CM_Filter[2][2] * frame[2][2]);
			}
		}
		reinterpret_cast<imtype2*>(&output[(tx * cols + ty)])[0] = make_imtype2(vals[0], vals[1]);
	}
}

#if !defined(INSUFFICIENT_MEMORY_FOR_CF16)
__global__ void SM_3x3_CF16(const imtype* __restrict__ input, imtype* __restrict__ output, const int rows, const int cols)
{
	__shared__  imtype cache[BLOCKDIMY + 2][BLOCKDIMX * 16 + 2];

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
			cache[0][cy + 12] = input[((tx - 1) * cols + ty + 12)];
			cache[0][cy + 13] = input[((tx - 1) * cols + ty + 13)];
			cache[0][cy + 14] = input[((tx - 1) * cols + ty + 14)];
			cache[0][cy + 15] = input[((tx - 1) * cols + ty + 15)];
		}
		if (cx == BLOCKDIMY) {
			cache[BLOCKDIMY + 1][cy] = input[((tx + 1) * cols + ty)];
			cache[BLOCKDIMY + 1][cy + 1] = input[((tx + 1) * cols + ty + 1)];
			cache[BLOCKDIMY + 1][cy + 2] = input[((tx + 1) * cols + ty + 2)];
			cache[BLOCKDIMY + 1][cy + 3] = input[((tx + 1) * cols + ty + 3)];
			cache[BLOCKDIMY + 1][cy + 4] = input[((tx + 1) * cols + ty + 4)];
			cache[BLOCKDIMY + 1][cy + 5] = input[((tx + 1) * cols + ty + 5)];
			cache[BLOCKDIMY + 1][cy + 6] = input[((tx + 1) * cols + ty + 6)];
			cache[BLOCKDIMY + 1][cy + 7] = input[((tx + 1) * cols + ty + 7)];
			cache[BLOCKDIMY + 1][cy + 8] = input[((tx + 1) * cols + ty + 8)];
			cache[BLOCKDIMY + 1][cy + 9] = input[((tx + 1) * cols + ty + 9)];
			cache[BLOCKDIMY + 1][cy + 10] = input[((tx + 1) * cols + ty + 10)];
			cache[BLOCKDIMY + 1][cy + 11] = input[((tx + 1) * cols + ty + 11)];
			cache[BLOCKDIMY + 1][cy + 12] = input[((tx + 1) * cols + ty + 12)];
			cache[BLOCKDIMY + 1][cy + 13] = input[((tx + 1) * cols + ty + 13)];
			cache[BLOCKDIMY + 1][cy + 14] = input[((tx + 1) * cols + ty + 14)];
			cache[BLOCKDIMY + 1][cy + 15] = input[((tx + 1) * cols + ty + 15)];
		}
		if (cy == 1) {
			cache[cx][0] = input[((tx)*cols + ty - 1)];
		}
		if (cy == BLOCKDIMX * 16 - 15) {
			cache[cx][BLOCKDIMX * 16 + 1] = input[((tx)*cols + ty + 16)];
		}

		__syncthreads();

		imtype frame[3][3];

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
		+ GM_Filter[2][2] * frame[2][2]);

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
				+ GM_Filter[2][2] * frame[2][2]);
			}
		}
	}
}
__global__ void SM_3x3_CF16_Vec(unsigned char* __restrict__ input, imtype* __restrict__ output, const int rows, const int cols)
{
	__shared__  imtype cache[BLOCKDIMY + 2][BLOCKDIMX * 16 + 2];

	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 16;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	int cy = threadIdx.x * 16 + 1;
	int cx = threadIdx.y + 1;

	imtype4 vec4;

	if ((tx > 0 && tx < rows - 1) && (ty < cols - 1)) {
		vec4 = reinterpret_cast<imtype4*>(&input[tx * cols + ty])[0];
		cache[cx][cy] = vec4.x;
		cache[cx][cy + 1] = vec4.y;
		cache[cx][cy + 2] = vec4.z;
		cache[cx][cy + 3] = vec4.w;

		vec4 = reinterpret_cast<imtype4*>(&input[tx * cols + ty + 4])[0];
		cache[cx][cy + 4] = vec4.x;
		cache[cx][cy + 5] = vec4.y;
		cache[cx][cy + 6] = vec4.z;
		cache[cx][cy + 7] = vec4.w;

		vec4 = reinterpret_cast<imtype4*>(&input[tx * cols + ty + 8])[0];
		cache[cx][cy + 8] = vec4.x;
		cache[cx][cy + 9] = vec4.y;
		cache[cx][cy + 10] = vec4.z;
		cache[cx][cy + 11] = vec4.w;

		vec4 = reinterpret_cast<imtype4*>(&input[tx * cols + ty + 12])[0];
		cache[cx][cy + 12] = vec4.x;
		cache[cx][cy + 13] = vec4.y;
		cache[cx][cy + 14] = vec4.z;
		cache[cx][cy + 15] = vec4.w;

		if (cx == 1) {
			vec4 = reinterpret_cast<imtype4*>(&input[(tx - 1) * cols + ty])[0];
			cache[0][cy] = vec4.x;
			cache[0][cy + 1] = vec4.y;
			cache[0][cy + 2] = vec4.z;
			cache[0][cy + 3] = vec4.w;

			vec4 = reinterpret_cast<imtype4*>(&input[(tx - 1) * cols + ty + 4])[0];
			cache[0][cy + 4] = vec4.x;
			cache[0][cy + 5] = vec4.y;
			cache[0][cy + 6] = vec4.z;
			cache[0][cy + 7] = vec4.w;

			vec4 = reinterpret_cast<imtype4*>(&input[(tx - 1) * cols + ty + 8])[0];
			cache[0][cy + 8] = vec4.x;
			cache[0][cy + 9] = vec4.y;
			cache[0][cy + 10] = vec4.z;
			cache[0][cy + 11] = vec4.w;

			vec4 = reinterpret_cast<imtype4*>(&input[(tx - 1) * cols + ty + 12])[0];
			cache[0][cy + 12] = vec4.x;
			cache[0][cy + 13] = vec4.y;
			cache[0][cy + 14] = vec4.z;
			cache[0][cy + 15] = vec4.w;
		}
		if (cx == BLOCKDIMY) {
			vec4 = reinterpret_cast<imtype4*>(&input[(tx + 1) * cols + ty])[0];
			cache[BLOCKDIMY + 1][cy] = vec4.x;
			cache[BLOCKDIMY + 1][cy + 1] = vec4.y;
			cache[BLOCKDIMY + 1][cy + 2] = vec4.z;
			cache[BLOCKDIMY + 1][cy + 3] = vec4.w;
			vec4 = reinterpret_cast<imtype4*>(&input[(tx + 1) * cols + ty + 4])[0];
			cache[BLOCKDIMY + 1][cy + 4] = vec4.x;
			cache[BLOCKDIMY + 1][cy + 5] = vec4.y;
			cache[BLOCKDIMY + 1][cy + 6] = vec4.z;
			cache[BLOCKDIMY + 1][cy + 7] = vec4.w;
			vec4 = reinterpret_cast<imtype4*>(&input[(tx + 1) * cols + ty + 8])[0];
			cache[BLOCKDIMY + 1][cy + 8] = vec4.x;
			cache[BLOCKDIMY + 1][cy + 9] = vec4.y;
			cache[BLOCKDIMY + 1][cy + 10] = vec4.z;
			cache[BLOCKDIMY + 1][cy + 11] = vec4.w;
			vec4 = reinterpret_cast<imtype4*>(&input[(tx + 1) * cols + ty + 12])[0];
			cache[BLOCKDIMY + 1][cy + 12] = vec4.x;
			cache[BLOCKDIMY + 1][cy + 13] = vec4.y;
			cache[BLOCKDIMY + 1][cy + 14] = vec4.z;
			cache[BLOCKDIMY + 1][cy + 15] = vec4.w;
		}
		if (cy == 1) {
			cache[cx][0] = input[((tx)*cols + ty - 1)];
		}
		if (cy == BLOCKDIMX * 16 - 15) {
			cache[cx][BLOCKDIMX * 16 + 1] = input[((tx)*cols + ty + 16)];
		}

		__syncthreads();
		imtype vals[16] = { 0 };
		imtype frame[3][3];

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
		+ GM_Filter[2][2] * frame[2][2]);

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
				+ GM_Filter[2][2] * frame[2][2]);
			}
		}
		reinterpret_cast<imtype4*>(&output[(tx * cols + ty)])[0] = make_imtype4(vals[0], vals[1], vals[2], vals[3]);
		reinterpret_cast<imtype4*>(&output[(tx * cols + ty + 4)])[0] = make_imtype4(vals[4], vals[5], vals[6], vals[7]);
		reinterpret_cast<imtype4*>(&output[(tx * cols + ty + 8)])[0] = make_imtype4(vals[8], vals[9], vals[10], vals[11]);
		reinterpret_cast<imtype4*>(&output[(tx * cols + ty + 12)])[0] = make_imtype4(vals[12], vals[13], vals[14], vals[15]);
	}
}
#endif


#if !defined(INSUFFICIENT_MEMORY_FOR_CF12)
__global__ void SM_3x3_CF12(const imtype* __restrict__ input, imtype *__restrict__ output, const int rows, const int cols)
{
	__shared__  imtype cache[BLOCKDIMY + 2][BLOCKDIMX * 12 + 2];

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
		if (cx == BLOCKDIMY) {
			cache[BLOCKDIMY + 1][cy] = input[((tx + 1) * cols + ty)];
			cache[BLOCKDIMY + 1][cy + 1] = input[((tx + 1) * cols + ty + 1)];
			cache[BLOCKDIMY + 1][cy + 2] = input[((tx + 1) * cols + ty + 2)];
			cache[BLOCKDIMY + 1][cy + 3] = input[((tx + 1) * cols + ty + 3)];
			cache[BLOCKDIMY + 1][cy + 4] = input[((tx + 1) * cols + ty + 4)];
			cache[BLOCKDIMY + 1][cy + 5] = input[((tx + 1) * cols + ty + 5)];
			cache[BLOCKDIMY + 1][cy + 6] = input[((tx + 1) * cols + ty + 6)];
			cache[BLOCKDIMY + 1][cy + 7] = input[((tx + 1) * cols + ty + 7)];
			cache[BLOCKDIMY + 1][cy + 8] = input[((tx + 1) * cols + ty + 8)];
			cache[BLOCKDIMY + 1][cy + 9] = input[((tx + 1) * cols + ty + 9)];
			cache[BLOCKDIMY + 1][cy + 10] = input[((tx + 1) * cols + ty + 10)];
			cache[BLOCKDIMY + 1][cy + 11] = input[((tx + 1) * cols + ty + 11)];
		}
		if (cy == 1) {
			cache[cx][0] = input[((tx)*cols + ty - 1)];
		}
		if (cy == BLOCKDIMX * 12 - 11) {
			cache[cx][BLOCKDIMX * 12 + 1] = input[((tx)*cols + ty + 12)];
		}

		__syncthreads();

		imtype frame[3][3];

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
		+ GM_Filter[2][2] * frame[2][2]);

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
				+ GM_Filter[2][2] * frame[2][2]);
			}
		}
	}
}

__global__ void SM_3x3_CF12_Vec(unsigned char* __restrict__ input, imtype* __restrict__ output, const int rows, const int cols)
{
	__shared__  imtype cache[BLOCKDIMY + 2][BLOCKDIMX * 12 + 2];

	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 12;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	int cy = threadIdx.x * 12 + 1;
	int cx = threadIdx.y + 1;
	
	imtype4 vec4;

	if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1)) {
		vec4 = reinterpret_cast<imtype4*>(&input[tx * cols + ty])[0];
		cache[cx][cy] = vec4.x;
		cache[cx][cy + 1] = vec4.y;
		cache[cx][cy + 2] = vec4.z;
		cache[cx][cy + 3] = vec4.w;

		vec4 = reinterpret_cast<imtype4*>(&input[tx * cols + ty + 4])[0];
		cache[cx][cy + 4] = vec4.x;
		cache[cx][cy + 5] = vec4.y;
		cache[cx][cy + 6] = vec4.z;
		cache[cx][cy + 7] = vec4.w;

		vec4 = reinterpret_cast<imtype4*>(&input[tx * cols + ty + 8])[0];
		cache[cx][cy + 8] = vec4.x;
		cache[cx][cy + 9] = vec4.y;
		cache[cx][cy + 10] = vec4.z;
		cache[cx][cy + 11] = vec4.w;
		if (cx == 1) {
			vec4 = reinterpret_cast<imtype4*>(&input[(tx - 1 ) * cols + ty])[0];
			cache[0][cy] = vec4.x;
			cache[0][cy + 1] = vec4.y;
			cache[0][cy + 2] = vec4.z;
			cache[0][cy + 3] = vec4.w;

			vec4 = reinterpret_cast<imtype4*>(&input[(tx - 1) * cols + ty + 4])[0];
			cache[0][cy + 4] = vec4.x;
			cache[0][cy + 5] = vec4.y;
			cache[0][cy + 6] = vec4.z;
			cache[0][cy + 7] = vec4.w;

			vec4 = reinterpret_cast<imtype4*>(&input[(tx - 1) * cols + ty + 8])[0];
			cache[0][cy + 8] = vec4.x;
			cache[0][cy + 9] = vec4.y;
			cache[0][cy + 10] = vec4.z; 
			cache[0][cy + 11] = vec4.w; 
		}
		if (cx == BLOCKDIMY) {
			vec4 = reinterpret_cast<imtype4*>(&input[(tx + 1) * cols + ty])[0];
			cache[BLOCKDIMY + 1][cy] = vec4.x;
			cache[BLOCKDIMY + 1][cy + 1] = vec4.y;
			cache[BLOCKDIMY + 1][cy + 2] = vec4.z;
			cache[BLOCKDIMY + 1][cy + 3] = vec4.w;
			vec4 = reinterpret_cast<imtype4*>(&input[(tx + 1) * cols + ty + 4])[0];
			cache[BLOCKDIMY + 1][cy + 4] = vec4.x;
			cache[BLOCKDIMY + 1][cy + 5] = vec4.y;
			cache[BLOCKDIMY + 1][cy + 6] = vec4.z;
			cache[BLOCKDIMY + 1][cy + 7] = vec4.w;
			vec4 = reinterpret_cast<imtype4*>(&input[(tx + 1) * cols + ty + 8])[0];
			cache[BLOCKDIMY + 1][cy + 8] = vec4.x;
			cache[BLOCKDIMY + 1][cy + 9] = vec4.y;
			cache[BLOCKDIMY + 1][cy + 10] = vec4.z;
			cache[BLOCKDIMY + 1][cy + 11] = vec4.w;
		}
		if (cy == 1) {
			cache[cx][0] = input[((tx)*cols + ty - 1)];
		}
		if (cy == BLOCKDIMX * 12 - 11) {
			cache[cx][BLOCKDIMX * 12 + 1] = input[((tx)*cols + ty + 12)];
		}
		__syncthreads();

		imtype vals[12] = { 0 };
		imtype frame[3][3];

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
		+ GM_Filter[2][2] * frame[2][2]);

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
				+ GM_Filter[2][2] * frame[2][2]);
			}
		}
		reinterpret_cast<imtype4*>(&output[(tx * cols + ty)])[0] = make_imtype4(vals[0], vals[1], vals[2], vals[3]);
		reinterpret_cast<imtype4*>(&output[(tx * cols + ty + 4)])[0] = make_imtype4(vals[4], vals[5], vals[6], vals[7]);
		reinterpret_cast<imtype4*>(&output[(tx * cols + ty + 8)])[0] = make_imtype4(vals[8], vals[9], vals[10], vals[11]);
	}
}
#endif

#if !defined(INSUFFICIENT_MEMORY_FOR_CF8)
__global__ void SM_3x3_CF8(const imtype* __restrict__ input, imtype* __restrict__ output, const int rows, const int cols)
{
	__shared__  imtype cache[BLOCKDIMY + 2][BLOCKDIMX * 8 + 2];

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
		if (cx == BLOCKDIMY) {
			cache[BLOCKDIMY + 1][cy] = input[((tx + 1) * cols + ty)];
			cache[BLOCKDIMY + 1][cy + 1] = input[((tx + 1) * cols + ty + 1)];
			cache[BLOCKDIMY + 1][cy + 2] = input[((tx + 1) * cols + ty + 2)];
			cache[BLOCKDIMY + 1][cy + 3] = input[((tx + 1) * cols + ty + 3)];
			cache[BLOCKDIMY + 1][cy + 4] = input[((tx + 1) * cols + ty + 4)];
			cache[BLOCKDIMY + 1][cy + 5] = input[((tx + 1) * cols + ty + 5)];
			cache[BLOCKDIMY + 1][cy + 6] = input[((tx + 1) * cols + ty + 6)];
			cache[BLOCKDIMY + 1][cy + 7] = input[((tx + 1) * cols + ty + 7)];
		}
		if (cy == 1) {
			cache[cx][0] = input[((tx)*cols + ty - 1)];
		}
		if (cy == BLOCKDIMX * 8 - 7) {
			cache[cx][BLOCKDIMX * 8 + 1] = input[((tx)*cols + ty + 8)];
		}
		__syncthreads();

		imtype frame[3][3];

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
		+ GM_Filter[2][2] * frame[2][2]);

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
				+ GM_Filter[2][2] * frame[2][2]);
			}
		}
	}
}

__global__ void SM_3x3_CF8_Vec(imtype* __restrict__ input, imtype* __restrict__ output, const int rows, const int cols)
{
	__shared__  imtype cache[BLOCKDIMY + 2][BLOCKDIMX * 8 + 2];

	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	int cy = threadIdx.x * 8 + 1;
	int cx = threadIdx.y + 1;

	imtype4 vec4;

	if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1)) {
		vec4 = reinterpret_cast<imtype4*>(&input[tx * cols + ty])[0];
		cache[cx][cy] = vec4.x;
		cache[cx][cy + 1] = vec4.y;
		cache[cx][cy + 2] = vec4.z;
		cache[cx][cy + 3] = vec4.w;

		vec4 = reinterpret_cast<imtype4*>(&input[tx * cols + ty + 4])[0];
		cache[cx][cy + 4] = vec4.x;
		cache[cx][cy + 5] = vec4.y;
		cache[cx][cy + 6] = vec4.z;
		cache[cx][cy + 7] = vec4.w;
		if (cx == 1) {
			vec4 = reinterpret_cast<imtype4*>(&input[(tx - 1) * cols + ty])[0];
			cache[0][cy] = vec4.x;
			cache[0][cy + 1] = vec4.y;
			cache[0][cy + 2] = vec4.z;
			cache[0][cy + 3] = vec4.w;
			vec4 = reinterpret_cast<imtype4*>(&input[(tx - 1) * cols + ty + 4])[0];
			cache[0][cy + 4] = vec4.x;
			cache[0][cy + 5] = vec4.y;
			cache[0][cy + 6] = vec4.z;
			cache[0][cy + 7] = vec4.w;
		}
		if (cx == BLOCKDIMY) {
			vec4 = reinterpret_cast<imtype4*>(&input[(tx + 1) * cols + ty])[0];
			cache[BLOCKDIMY + 1][cy] = vec4.x;
			cache[BLOCKDIMY + 1][cy + 1] = vec4.y;
			cache[BLOCKDIMY + 1][cy + 2] = vec4.z;
			cache[BLOCKDIMY + 1][cy + 3] = vec4.w;
			vec4 = reinterpret_cast<imtype4*>(&input[(tx + 1) * cols + ty + 4])[0];
			cache[BLOCKDIMY + 1][cy + 4] = vec4.x;
			cache[BLOCKDIMY + 1][cy + 5] = vec4.y;
			cache[BLOCKDIMY + 1][cy + 6] = vec4.z;
			cache[BLOCKDIMY + 1][cy + 7] = vec4.w;
		}
		if (cy == 1) {
			cache[cx][0] = input[((tx)*cols + ty - 1)];
		}
		if (cy == BLOCKDIMX * 8 - 7) {
			cache[cx][BLOCKDIMX * 8 + 1] = input[(tx*cols + ty + 8)];
		}
		__syncthreads();
		imtype vals[8] = { 0 };
		imtype frame[3][3];

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
		+ GM_Filter[2][2] * frame[2][2]);

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
				+ GM_Filter[2][2] * frame[2][2]);
			}
		}
		reinterpret_cast<imtype4*>(&output[(tx * cols + ty)])[0] = make_imtype4(vals[0], vals[1], vals[2], vals[3]);
		reinterpret_cast<imtype4*>(&output[(tx * cols + ty + 4)])[0] = make_imtype4(vals[4], vals[5], vals[6], vals[7]);
	}
}
#endif

#if !defined(INSUFFICIENT_MEMORY_FOR_CF4)
__global__ void SM_3x3_CF4(const imtype* __restrict__ input, imtype* __restrict__ output, const int rows, const int cols)
{
	__shared__  imtype cache[BLOCKDIMY + 2][BLOCKDIMX * 4 + 2];

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
		if (cx == BLOCKDIMY) {
			cache[BLOCKDIMY + 1][cy] = input[((tx + 1) * cols + ty)];
			cache[BLOCKDIMY + 1][cy + 1] = input[((tx + 1) * cols + ty + 1)];
			cache[BLOCKDIMY + 1][cy + 2] = input[((tx + 1) * cols + ty + 2)];
			cache[BLOCKDIMY + 1][cy + 3] = input[((tx + 1) * cols + ty + 3)];
		}
		if (cy == 1) {
			cache[cx][0] = input[((tx)*cols + ty - 1)];
		}
		if (cy == BLOCKDIMX * 4 - 3) {
			cache[cx][BLOCKDIMX * 4 + 1] = input[((tx)*cols + ty + 4)];
		}
		__syncthreads();

		imtype frame[3][3];

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
		+ GM_Filter[2][2] * frame[2][2]);

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
				+ GM_Filter[2][2] * frame[2][2]);
			}
		}
	}
}
__global__ void SM_3x3_CF4_Vec(imtype* __restrict__ input, imtype* __restrict__ output, const int rows, const int cols)
{
	__shared__  imtype cache[BLOCKDIMY + 2][BLOCKDIMX * 4 + 2];

	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	int cy = threadIdx.x * 4 + 1;
	int cx = threadIdx.y + 1;

	imtype4 vec4;

	if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1)) {
		vec4 = reinterpret_cast<imtype4*>(&input[tx * cols + ty])[0];
		cache[cx][cy] = vec4.x;
		cache[cx][cy + 1] = vec4.y;
		cache[cx][cy + 2] = vec4.z;
		cache[cx][cy + 3] = vec4.w;
		if (cx == 1) {
			vec4 = reinterpret_cast<imtype4*>(&input[(tx - 1) * cols + ty])[0];
			cache[0][cy] = vec4.x;
			cache[0][cy + 1] = vec4.y;
			cache[0][cy + 2] = vec4.z;
			cache[0][cy + 3] = vec4.w;
		}
		if (cx == BLOCKDIMY) {
			vec4 = reinterpret_cast<imtype4*>(&input[(tx + 1) * cols + ty])[0];
			cache[BLOCKDIMY + 1][cy] = vec4.x;
			cache[BLOCKDIMY + 1][cy + 1] = vec4.y;
			cache[BLOCKDIMY + 1][cy + 2] = vec4.z;
			cache[BLOCKDIMY + 1][cy + 3] = vec4.w;
		}
		if (cy == 1) {
			cache[cx][0] = input[(tx * cols + ty - 1)];
		}
		if (cy == BLOCKDIMX * 4 - 3) {
			cache[cx][BLOCKDIMX * 4 + 1] = input[(tx * cols + ty + 4)];
		}
		__syncthreads();

		imtype vals[4] = { 0 };
		imtype frame[3][3];

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
		+ GM_Filter[2][2] * frame[2][2]);

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
				+ GM_Filter[2][2] * frame[2][2]);
			}
		}
		reinterpret_cast<imtype4*>(&output[(tx * cols + ty)])[0] = make_imtype4(vals[0], vals[1], vals[2], vals[3]);
	}
}
#endif

#if !defined(INSUFFICIENT_MEMORY_FOR_CF2)
__global__ void SM_3x3_CF2(const imtype* __restrict__ input, imtype* __restrict__ output, const int rows, const int cols)
{
	__shared__  imtype cache[BLOCKDIMY + 2][BLOCKDIMX * 2 + 2];

	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	int cy = threadIdx.x * 2 + 1;
	int cx = threadIdx.y + 1;

	if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1)) {
		cache[cx][cy] = input[tx * cols + ty];
		cache[cx][cy + 1] = input[tx * cols + ty + 1];

		if (cx == 1) {
			cache[0][cy] = input[((tx - 1) * cols + ty)];
			cache[0][cy + 1] = input[((tx - 1) * cols + ty + 1)];
		}
		if (cx == BLOCKDIMY) {
			cache[BLOCKDIMY + 1][cy] = input[((tx + 1) * cols + ty)];
			cache[BLOCKDIMY + 1][cy + 1] = input[((tx + 1) * cols + ty + 1)];
		}
		if (cy == 1) {
			cache[cx][0] = input[(tx*cols + ty - 1)];
		}
		if (cy == BLOCKDIMX * 2 - 1) {
			cache[cx][BLOCKDIMX * 2 + 1] = input[(tx*cols + ty + 2)];
		}
		__syncthreads();

		imtype frame[3][3];

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
		+ GM_Filter[2][2] * frame[2][2]);

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
				+ GM_Filter[2][2] * frame[2][2]);
			}
		}
	}
}
__global__ void SM_3x3_CF2_Vec(imtype* __restrict__ input, imtype* __restrict__ output, const int rows, const int cols)
{
	__shared__  imtype cache[BLOCKDIMY + 2][BLOCKDIMX * 2 + 2];

	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	int cy = threadIdx.x * 2 + 1;
	int cx = threadIdx.y + 1;

	imtype2 vec;

	if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1)) {
		vec = reinterpret_cast<imtype2*>(&input[tx * cols + ty])[0];
		cache[cx][cy] = vec.x;
		cache[cx][cy + 1] = vec.y;

		if (cx == 1) {
			vec = reinterpret_cast<imtype2*>(&input[(tx - 1) * cols + ty])[0];
			cache[0][cy] = vec.x;
			cache[0][cy + 1] = vec.y;
		}
		if (cx == BLOCKDIMY) {
			vec = reinterpret_cast<imtype2*>(&input[(tx + 1) * cols + ty])[0];
			cache[BLOCKDIMY + 1][cy] = vec.x;
			cache[BLOCKDIMY + 1][cy + 1] = vec.y;
		}
		if (cy == 1) {
			cache[cx][0] = input[(tx * cols + ty - 1)];
		}
		if (cy == BLOCKDIMX * 2 - 1) {
			cache[cx][BLOCKDIMX * 2 + 1] = input[(tx * cols + ty + 2)];
		}
		__syncthreads();

		imtype vals[2] = { 0 };
		imtype frame[3][3];

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
		+ GM_Filter[2][2] * frame[2][2]);

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
					+ GM_Filter[2][2] * frame[2][2]);
			}
		}
		reinterpret_cast<imtype2*>(&output[(tx * cols + ty)])[0] = make_imtype2(vals[0], vals[1]);
	}
}
#endif

void saveImage(const void* data, int rows, int cols, const std::string& filename) {
    cv::Mat image(rows, cols, IMAGE_TYPE, const_cast<void*>(data));

#ifdef IMTYPE_FLOAT
    cv::Mat output_image;
    cv::normalize(image, output_image, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::imwrite(filename, output_image);
#else
    cv::imwrite(filename, image);
#endif
}

void test_outputs(cv::Mat* input_img, cv::Mat* output_img)
{
	imtype* d_input = nullptr;
	imtype* d_output = nullptr;
	imtype* h_input = reinterpret_cast<imtype*>(input_img->data);
	imtype* h_output = reinterpret_cast<imtype*>(output_img->data);

	const int cols = (*input_img).cols;
	const int rows = (*input_img).rows;
	const int size = cols * rows * sizeof(imtype);

	dim3 block(BLOCKDIMX,BLOCKDIMY);
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
	
	#define BASE_KERNELS 1
	#define COARSENED_KERNELS 1
	#define VECTORIZED_KERNELS 1
	#define NPP_KERNEL 0
	#define CUDNN_KERNEL 0
	#define ARRAYFIRE 0
	#define OPENCV_CUDA 0

	#if BASE_KERNELS
	GM_3x3 << <grid, block >> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	saveImage(h_output, rows, cols, "../images/outputs/GM_3x3.png");
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	CM_3x3 << <grid, block >> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	saveImage(h_output, rows, cols, "../images/outputs/CM_3x3.png");
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	SM_3x3 << <grid, block >> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	saveImage(h_output, rows, cols, "../images/outputs/SM_3x3.png");
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));
	#endif

	#if COARSENED_KERNELS
	GM_3x3_CF2 << <grid_cf2, block >> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	saveImage(h_output, rows, cols, "../images/outputs/GM_3x3_CF2.png");
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	GM_3x3_CF4 << <grid_cf4, block >> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	saveImage(h_output, rows, cols, "../images/outputs/GM_3x3_CF4.png");
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	GM_3x3_CF8 << <grid_cf8, block >> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	saveImage(h_output, rows, cols, "../images/outputs/GM_3x3_CF8.png");
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	GM_3x3_CF12 << <grid_cf12, block >> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	saveImage(h_output, rows, cols, "../images/outputs/GM_3x3_CF12.png");
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	GM_3x3_CF16 << <grid_cf16, block >> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	saveImage(h_output, rows, cols, "../images/outputs/GM_3x3_CF16.png");
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	CM_3x3_CF2 << <grid_cf2, block >> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	saveImage(h_output, rows, cols, "../images/outputs/CM_3x3_CF2.png");
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	CM_3x3_CF4 << <grid_cf4, block >> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	saveImage(h_output, rows, cols, "../images/outputs/CM_3x3_CF4.png");
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	CM_3x3_CF8 << <grid_cf8, block >> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	saveImage(h_output, rows, cols, "../images/outputs/CM_3x3_CF8.png");
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	CM_3x3_CF12 << <grid_cf12, block >> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	saveImage(h_output, rows, cols, "../images/outputs/CM_3x3_CF12.png");
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	CM_3x3_CF16 << <grid_cf16, block >> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	saveImage(h_output, rows, cols, "../images/outputs/CM_3x3_CF16.png");
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	SM_3x3_CF2 << <grid_cf2, block >> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	saveImage(h_output, rows, cols, "../images/outputs/SM_3x3_CF2.png");
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	SM_3x3_CF4 << <grid_cf4, block >> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	saveImage(h_output, rows, cols, "../images/outputs/SM_3x3_CF4.png");
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	SM_3x3_CF8 << <grid_cf8, block >> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	saveImage(h_output, rows, cols, "../images/outputs/SM_3x3_CF8.png");
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	#if !defined(INSUFFICIENT_MEMORY_FOR_CF12)
	SM_3x3_CF12 << <grid_cf12, block >> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	saveImage(h_output, rows, cols, "../images/outputs/SM_3x3_CF12.png");
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));
	#endif

	#if !defined(INSUFFICIENT_MEMORY_FOR_CF16)
	SM_3x3_CF16 << <grid_cf16, block >> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	saveImage(h_output, rows, cols, "../images/outputs/SM_3x3_CF16.png");
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));
	#endif
	
	#endif

	#if VECTORIZED_KERNELS
	GM_3x3_CF2_Vec << <grid_cf2, block>> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	saveImage(h_output, rows, cols, "../images/outputs/GM_3x3_CF2_Vec.png");
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	GM_3x3_CF4_Vec << <grid_cf4, block>> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	saveImage(h_output, rows, cols, "../images/outputs/GM_3x3_CF4_Vec.png");
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	GM_3x3_CF8_Vec << <grid_cf8, block>> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	saveImage(h_output, rows, cols, "../images/outputs/GM_3x3_CF8_Vec.png");
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	GM_3x3_CF12_Vec << <grid_cf12, block>> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	saveImage(h_output, rows, cols, "../images/outputs/GM_3x3_CF12_Vec.png");
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	GM_3x3_CF16_Vec << <grid_cf16, block>> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	saveImage(h_output, rows, cols, "../images/outputs/GM_3x3_CF16_Vec.png");
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	CM_3x3_CF2_Vec << <grid_cf2, block>> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	saveImage(h_output, rows, cols, "../images/outputs/CM_3x3_CF2_Vec.png");
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	CM_3x3_CF4_Vec << <grid_cf4, block>> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	saveImage(h_output, rows, cols, "../images/outputs/CM_3x3_CF4_Vec.png");
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	CM_3x3_CF8_Vec << <grid_cf8, block>> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	saveImage(h_output, rows, cols, "../images/outputs/CM_3x3_CF8_Vec.png");
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	CM_3x3_CF12_Vec << <grid_cf12, block>> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	saveImage(h_output, rows, cols, "../images/outputs/CM_3x3_CF12_Vec.png");
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	CM_3x3_CF16_Vec << <grid_cf16, block>> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	saveImage(h_output, rows, cols, "../images/outputs/CM_3x3_CF16_Vec.png");
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	SM_3x3_CF2_Vec << <grid_cf2, block>> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	saveImage(h_output, rows, cols, "../images/outputs/SM_3x3_CF2_Vec.png");
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	SM_3x3_CF4_Vec << <grid_cf4, block>> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	saveImage(h_output, rows, cols, "../images/outputs/SM_3x3_CF4_Vec.png");
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	SM_3x3_CF8_Vec << <grid_cf8, block>> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	saveImage(h_output, rows, cols, "../images/outputs/SM_3x3_CF8_Vec.png");
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	#if !defined(INSUFFICIENT_MEMORY_FOR_CF12)
	SM_3x3_CF12_Vec << <grid_cf12, block>> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	saveImage(h_output, rows, cols, "../images/outputs/SM_3x3_CF12_Vec.png");
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));
	#endif

	#if !defined(INSUFFICIENT_MEMORY_FOR_CF16)
	SM_3x3_CF16_Vec << <grid_cf16, block>> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	saveImage(h_output, rows, cols, "../images/outputs/SM_3x3_CF16_Vec.png");
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));
	#endif
	
	#endif

	#if NPP_KERNEL && defined(IMTYPE_FLOAT)
	{
		CHECK_NPP_ERROR(nppiFilterGauss_32f_C1R(d_input, input_img->step, d_output, input_img->step, {cols, rows}, NPP_MASK_SIZE_3_X_3));
		CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
		saveImage(h_output, rows, cols, "../images/outputs/nppiFilterGauss_32f_C1R.png");
		CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

		const Npp32f h_kernel[9] = {1/16.0f, 2/16.0f, 1/16.0f,
									2/16.0f, 4/16.0f, 2/16.0f,
									1/16.0f, 2/16.0f, 1/16.0f};
		Npp32f* d_kernel = NULL;

		CHECK_CUDA_ERROR(cudaMalloc(&d_kernel, sizeof(Npp32f) * 9));
		CHECK_CUDA_ERROR(cudaMemcpy(d_kernel, h_kernel, sizeof(Npp32f) * 9, cudaMemcpyHostToDevice));
		CHECK_NPP_ERROR(nppiFilter_32f_C1R(d_input, cols * sizeof(imtype), d_output, cols * sizeof(imtype), {cols, rows}, d_kernel, {3,3}, {1,1}));
		CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
		saveImage(h_output, rows, cols, "../images/outputs/nppiFilter_32f_C1R.png");
		CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));
	}
	#endif

	#if CUDNN_KERNEL && defined(IMTYPE_FLOAT)
	{
		cudnnHandle_t cudnn;
		CHECK_CUDNN_ERROR(cudnnCreate(&cudnn));

		cudnnTensorDescriptor_t input_descriptor, output_descriptor;
		CHECK_CUDNN_ERROR(cudnnCreateTensorDescriptor(&input_descriptor));
		CHECK_CUDNN_ERROR(cudnnSetTensor4dDescriptor(input_descriptor, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, 1, 1, rows, cols));
		CHECK_CUDNN_ERROR(cudnnCreateTensorDescriptor(&output_descriptor));
		CHECK_CUDNN_ERROR(cudnnSetTensor4dDescriptor(output_descriptor, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, 1, 1, rows, cols));

		cudnnFilterDescriptor_t f_descriptor;
		CHECK_CUDNN_ERROR(cudnnCreateFilterDescriptor(&f_descriptor));
		CHECK_CUDNN_ERROR(cudnnSetFilter4dDescriptor(f_descriptor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NHWC, 1, 1, 3, 3));

		cudnnConvolutionDescriptor_t conv_descriptor;
		CHECK_CUDNN_ERROR(cudnnCreateConvolutionDescriptor(&conv_descriptor));
		CHECK_CUDNN_ERROR(cudnnSetConvolution2dDescriptor(conv_descriptor, 1, 1, 1, 1, 1, 1, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));

		size_t workspace_size = 0;
		void* d_workspace = nullptr;
		
		CHECK_CUDNN_ERROR(cudnnGetConvolutionForwardWorkspaceSize(
			cudnn, input_descriptor, f_descriptor, conv_descriptor, output_descriptor, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, &workspace_size));

		std::cout << workspace_size << std::endl;
		CHECK_CUDA_ERROR(cudaMalloc(&d_workspace, workspace_size));

		float alpha = 1.0f;
		float beta = 0.0f;
		const float h_kernel[9] = {1 / 16.0f, 2 / 16.0f, 1 / 16.0f, 2 / 16.0f, 4 / 16.0f, 2 / 16.0f, 1 / 16.0f, 2 / 16.0f, 1 / 16.0f};
		float* d_kernel = NULL;
		CHECK_CUDA_ERROR(cudaMalloc(&d_kernel, sizeof(float) * 9));
		CHECK_CUDA_ERROR(cudaMemcpy(d_kernel, h_kernel, sizeof(float) * 9, cudaMemcpyHostToDevice));

		CHECK_CUDNN_ERROR(cudnnConvolutionForward(cudnn, &alpha, input_descriptor, d_input, f_descriptor, d_kernel, conv_descriptor,
			CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, d_workspace, workspace_size, &beta, output_descriptor, d_output));

		CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
			saveImage(h_output, rows, cols, "../images/outputs/cuDNN.png");
		CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

		CHECK_CUDNN_ERROR(cudnnDestroyTensorDescriptor(input_descriptor));
		CHECK_CUDNN_ERROR(cudnnDestroyTensorDescriptor(output_descriptor));
		CHECK_CUDNN_ERROR(cudnnDestroyFilterDescriptor(f_descriptor));
		CHECK_CUDNN_ERROR(cudnnDestroyConvolutionDescriptor(conv_descriptor));
		CHECK_CUDNN_ERROR(cudnnDestroy(cudnn));
	}
	#endif

	#if ARRAYFIRE
	{
		CHECK_ARRAYFIRE(af_init());
		CHECK_ARRAYFIRE(af_set_backend(AF_BACKEND_CUDA));

		af_backend active_backend;
		af_get_active_backend(&active_backend);
		if (active_backend != AF_BACKEND_CUDA) {
			std::cerr << "CUDA backend is inactive.. abort()" << active_backend << std::endl;
			abort();
		}

		dim_t dims_of_input[2] = {rows, cols};
		dim_t dims_of_kernel[2] = {3, 3};

		af_array af_input, af_kernel, af_output;

		#ifdef IMTYPE_FLOAT
		imtype h_kernel[9] = {1/16.0f, 2/16.0f, 1/16.0f,
					2/16.0f, 4/16.0f, 2/16.0f,
					1/16.0f, 2/16.0f, 1/16.0f};
		CHECK_ARRAYFIRE(af_create_array(&af_input, h_input, 2, dims_of_input, f32));
		CHECK_ARRAYFIRE(af_create_array(&af_kernel, h_kernel, 2, dims_of_kernel, f32));			
		#elif defined(IMTYPE_UCHAR)
		float h_kernel[9] = {1/16.0f, 2/16.0f, 1/16.0f,
			2/16.0f, 4/16.0f, 2/16.0f,
			1/16.0f, 2/16.0f, 1/16.0f};
		CHECK_ARRAYFIRE(af_create_array(&af_input, h_input, 2, dims_of_input, u8));
		CHECK_ARRAYFIRE(af_create_array(&af_kernel, h_kernel, 2, dims_of_kernel, f32));		
		#endif
		CHECK_ARRAYFIRE(af_convolve2(&af_output, af_input, af_kernel, AF_CONV_DEFAULT, AF_CONV_AUTO));
		CHECK_ARRAYFIRE(af_get_data_ptr(h_output, af_output));
		saveImage(h_output, rows, cols, "../images/outputs/arrayFire.png");
	}
	#endif

	#if OPENCV_CUDA
	{
		try
		{
			cv::cuda::GpuMat d_input, d_output;
			d_input.upload(*input_img);
			d_output.upload(*output_img);

			cv::Mat h_kernel = (cv::Mat_<float>(3, 3) << 
						1/16.0f, 2/16.0f, 1/16.0f,
						2/16.0f, 4/16.0f, 2/16.0f,
						1/16.0f, 2/16.0f, 1/16.0f);
			
			cv::Ptr<cv::cuda::Filter> gaussianFilter = cv::cuda::createLinearFilter(d_input.type(), d_output.type(), h_kernel);
			gaussianFilter->apply(d_input, d_output);

			cv::Mat blurred;
			d_output.download(blurred);
			saveImage(blurred.data, rows, cols, "../images/outputs/openCV-createLinearFilter.png");

			gaussianFilter = cv::cuda::createGaussianFilter(d_input.type(), d_output.type(), cv::Size(3,3), 1.0f);
			gaussianFilter->apply(d_input, d_output);

			d_output.download(blurred);
			saveImage(blurred.data, rows, cols, "../images/outputs/openCV-createGaussianFilter.png");
		}
		catch(const std::exception& e)
		{
			std::cerr << e.what() << '\n';
		}
		
	}
	#endif

	CHECK_CUDA_ERROR(cudaHostUnregister(h_input));
	CHECK_CUDA_ERROR(cudaHostUnregister(h_output));
	CHECK_CUDA_ERROR(cudaFree(d_input));
	CHECK_CUDA_ERROR(cudaFree(d_output));
}

void call_kernel(cv::Mat* input_img, cv::Mat* output_img, kernels::kernel func){
	imtype* d_input = nullptr;
	imtype* d_output = nullptr;
	imtype* h_input = reinterpret_cast<imtype*>(input_img->data);
	imtype* h_output = reinterpret_cast<imtype*>(output_img->data);

	const int cols = (*input_img).cols;
	const int rows = (*input_img).rows;
	const int size = cols * rows * sizeof(imtype);

	dim3 block(BLOCKDIMX,BLOCKDIMY);
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

	switch (func)
	{
	case kernels::kernel::GM_3x3:
		GM_3x3 << <grid, block >> > (d_input, d_output, rows, cols);
		break;
	case kernels::kernel::CM_3x3:
		CM_3x3 << <grid, block >> > (d_input, d_output, rows, cols);
		break;
	case kernels::kernel::SM_3x3:
		SM_3x3 << <grid, block >> > (d_input, d_output, rows, cols);
		break;
	case kernels::kernel::GM_3x3_CF2:
		GM_3x3_CF2 << <grid_cf2, block >> > (d_input, d_output, rows, cols);
		break;
	case kernels::kernel::GM_3x3_CF4:
		GM_3x3_CF4 << <grid_cf4, block >> > (d_input, d_output, rows, cols);
		break;
	case kernels::kernel::GM_3x3_CF8:
		GM_3x3_CF8 << <grid_cf8, block >> > (d_input, d_output, rows, cols);
		break;
	case kernels::kernel::GM_3x3_CF12:
		GM_3x3_CF12 << <grid_cf12, block >> > (d_input, d_output, rows, cols);
		break;
	case kernels::kernel::GM_3x3_CF16:
		GM_3x3_CF16 << <grid_cf16, block >> > (d_input, d_output, rows, cols);
		break;
	case kernels::kernel::CM_3x3_CF2:
		CM_3x3_CF2 << <grid_cf2, block >> > (d_input, d_output, rows, cols);
		break;
	case kernels::kernel::CM_3x3_CF4:
		CM_3x3_CF4 << <grid_cf4, block >> > (d_input, d_output, rows, cols);
		break;
	case kernels::kernel::CM_3x3_CF8:
		CM_3x3_CF8 << <grid_cf8, block >> > (d_input, d_output, rows, cols);
		break;
	case kernels::kernel::CM_3x3_CF12:
		CM_3x3_CF12 << <grid_cf12, block >> > (d_input, d_output, rows, cols);
		break;
	case kernels::kernel::CM_3x3_CF16:
		CM_3x3_CF16 << <grid_cf16, block >> > (d_input, d_output, rows, cols);
		break;
	case kernels::kernel::GM_3x3_CF2_Vec:
		GM_3x3_CF2_Vec << <grid_cf2, block >> > (d_input, d_output, rows, cols);
		break;
	case kernels::kernel::GM_3x3_CF4_Vec:
		GM_3x3_CF4_Vec << <grid_cf4, block >> > (d_input, d_output, rows, cols);
		break;
	case kernels::kernel::GM_3x3_CF8_Vec:
		GM_3x3_CF8_Vec << <grid_cf8, block >> > (d_input, d_output, rows, cols);
		break;
	case kernels::kernel::GM_3x3_CF12_Vec:
		GM_3x3_CF12_Vec << <grid_cf12, block >> > (d_input, d_output, rows, cols);
		break;
	case kernels::kernel::GM_3x3_CF16_Vec:
		GM_3x3_CF16_Vec << <grid_cf16, block >> > (d_input, d_output, rows, cols);
		break;
	case kernels::kernel::CM_3x3_CF2_Vec:
		CM_3x3_CF2_Vec << <grid_cf2, block >> > (d_input, d_output, rows, cols);
		break;
	case kernels::kernel::CM_3x3_CF4_Vec:
		CM_3x3_CF4_Vec << <grid_cf4, block >> > (d_input, d_output, rows, cols);
		break;
	case kernels::kernel::CM_3x3_CF8_Vec:
		CM_3x3_CF8_Vec << <grid_cf8, block >> > (d_input, d_output, rows, cols);
		break;
	case kernels::kernel::CM_3x3_CF12_Vec:
		CM_3x3_CF12_Vec << <grid_cf12, block >> > (d_input, d_output, rows, cols);
		break;
	case kernels::kernel::CM_3x3_CF16_Vec:
		CM_3x3_CF16_Vec << <grid_cf16, block >> > (d_input, d_output, rows, cols);
		break;
	case kernels::kernel::SM_3x3_CF2:
		SM_3x3_CF2 << <grid_cf2, block >> > (d_input, d_output, rows, cols);
		break;
	case kernels::kernel::SM_3x3_CF4:
		SM_3x3_CF4 << <grid_cf4, block >> > (d_input, d_output, rows, cols);
		break;
	case kernels::kernel::SM_3x3_CF8:
		SM_3x3_CF8 << <grid_cf8, block >> > (d_input, d_output, rows, cols);
		break;
	case kernels::kernel::SM_3x3_CF12:
		SM_3x3_CF12 << <grid_cf12, block >> > (d_input, d_output, rows, cols);
		break;
	case kernels::kernel::SM_3x3_CF16:
		SM_3x3_CF16 << <grid_cf16, block >> > (d_input, d_output, rows, cols);
		break;
	case kernels::kernel::SM_3x3_CF2_Vec:
		SM_3x3_CF2_Vec << <grid_cf2, block >> > (d_input, d_output, rows, cols);
		break;
	case kernels::kernel::SM_3x3_CF4_Vec:
		SM_3x3_CF4_Vec << <grid_cf4, block >> > (d_input, d_output, rows, cols);
		break;
	case kernels::kernel::SM_3x3_CF8_Vec:
		SM_3x3_CF8_Vec << <grid_cf8, block >> > (d_input, d_output, rows, cols);
		break;
	case kernels::kernel::SM_3x3_CF12_Vec:
		SM_3x3_CF12_Vec << <grid_cf12, block >> > (d_input, d_output, rows, cols);
		break;
	case kernels::kernel::SM_3x3_CF16_Vec:
		SM_3x3_CF16_Vec << <grid_cf16, block >> > (d_input, d_output, rows, cols);
		break;
	case kernels::kernel::ArrayFire:
			{
			CHECK_ARRAYFIRE(af_init());
			CHECK_ARRAYFIRE(af_set_backend(AF_BACKEND_CUDA));

			af_backend active_backend;
			af_get_active_backend(&active_backend);
			if (active_backend != AF_BACKEND_CUDA) {
				std::cerr << "CUDA backend is inactive.. abort()" << active_backend << std::endl;
				abort();
			}

			dim_t dims_of_input[2] = {rows, cols};
			dim_t dims_of_kernel[2] = {3, 3};

			af_array af_input, af_kernel, af_output;

			#ifdef IMTYPE_FLOAT
			imtype h_kernel[9] = {1/16.0f, 2/16.0f, 1/16.0f,
						2/16.0f, 4/16.0f, 2/16.0f,
						1/16.0f, 2/16.0f, 1/16.0f};
			CHECK_ARRAYFIRE(af_create_array(&af_input, h_input, 2, dims_of_input, f32));
			CHECK_ARRAYFIRE(af_create_array(&af_kernel, h_kernel, 2, dims_of_kernel, f32));			
			#elif defined(IMTYPE_UCHAR)
			float h_kernel[9] = {1/16.0f, 2/16.0f, 1/16.0f,
				2/16.0f, 4/16.0f, 2/16.0f,
				1/16.0f, 2/16.0f, 1/16.0f};
			CHECK_ARRAYFIRE(af_create_array(&af_input, h_input, 2, dims_of_input, u8));
			CHECK_ARRAYFIRE(af_create_array(&af_kernel, h_kernel, 2, dims_of_kernel, f32));		
			#endif
			CHECK_ARRAYFIRE(af_convolve2(&af_output, af_input, af_kernel, AF_CONV_DEFAULT, AF_CONV_AUTO));
			CHECK_ARRAYFIRE(af_get_data_ptr(h_output, af_output));
		}
		break;
	case kernels::kernel::OpenCV:
		{
			try
			{
				cv::cuda::GpuMat d_input, d_output;
				d_input.upload(*input_img);
				d_output.upload(*output_img);

				cv::Mat h_kernel = (cv::Mat_<float>(3, 3) << 
							1/16.0f, 2/16.0f, 1/16.0f,
							2/16.0f, 4/16.0f, 2/16.0f,
							1/16.0f, 2/16.0f, 1/16.0f);
				
				cv::Ptr<cv::cuda::Filter> gaussianFilter = cv::cuda::createLinearFilter(d_input.type(), d_output.type(), h_kernel);
				gaussianFilter->apply(d_input, d_output);

				cv::Mat blurred;
				d_output.download(blurred);

				gaussianFilter = cv::cuda::createGaussianFilter(d_input.type(), d_output.type(), cv::Size(3,3), 1.0f);
				gaussianFilter->apply(d_input, d_output);

				d_output.download(blurred);
			}
			catch(const std::exception& e)
			{
				std::cerr << e.what() << '\n';
			}
		}
		break;
	default:
		break;
	}
	CHECK_CUDA_ERROR(cudaHostUnregister(h_input));
	CHECK_CUDA_ERROR(cudaHostUnregister(h_output));
	CHECK_CUDA_ERROR(cudaFree(d_input));
	CHECK_CUDA_ERROR(cudaFree(d_output));
	CHECK_CUDA_ERROR(cudaDeviceReset());
}


void launch_kernels(cv::Mat* input_img, cv::Mat* output_img){
	call_kernel(input_img, output_img, kernels::kernel::GM_3x3);
	call_kernel(input_img, output_img, kernels::kernel::CM_3x3);
	call_kernel(input_img, output_img, kernels::kernel::SM_3x3);

	call_kernel(input_img, output_img, kernels::kernel::GM_3x3_CF2);
	call_kernel(input_img, output_img, kernels::kernel::GM_3x3_CF4);
	call_kernel(input_img, output_img, kernels::kernel::GM_3x3_CF8);
	call_kernel(input_img, output_img, kernels::kernel::GM_3x3_CF12);
	call_kernel(input_img, output_img, kernels::kernel::GM_3x3_CF16);

	call_kernel(input_img, output_img, kernels::kernel::CM_3x3_CF2);
	call_kernel(input_img, output_img, kernels::kernel::CM_3x3_CF4);
	call_kernel(input_img, output_img, kernels::kernel::CM_3x3_CF8);
	call_kernel(input_img, output_img, kernels::kernel::CM_3x3_CF12);
	call_kernel(input_img, output_img, kernels::kernel::CM_3x3_CF16);

	// call_kernel(input_img, output_img, kernels::kernel::SM_3x3_CF2);
	// call_kernel(input_img, output_img, kernels::kernel::SM_3x3_CF4);
	// call_kernel(input_img, output_img, kernels::kernel::SM_3x3_CF8);
	// call_kernel(input_img, output_img, kernels::kernel::SM_3x3_CF12);
	// call_kernel(input_img, output_img, kernels::kernel::SM_3x3_CF16);

	call_kernel(input_img, output_img, kernels::kernel::GM_3x3_CF2_Vec);
	call_kernel(input_img, output_img, kernels::kernel::GM_3x3_CF4_Vec);
	call_kernel(input_img, output_img, kernels::kernel::GM_3x3_CF8_Vec);
	call_kernel(input_img, output_img, kernels::kernel::GM_3x3_CF12_Vec);
	call_kernel(input_img, output_img, kernels::kernel::GM_3x3_CF16_Vec);

	call_kernel(input_img, output_img, kernels::kernel::CM_3x3_CF2_Vec);
	call_kernel(input_img, output_img, kernels::kernel::CM_3x3_CF4_Vec);
	call_kernel(input_img, output_img, kernels::kernel::CM_3x3_CF8_Vec);
	call_kernel(input_img, output_img, kernels::kernel::CM_3x3_CF12_Vec);
	call_kernel(input_img, output_img, kernels::kernel::CM_3x3_CF16_Vec);

	// call_kernel(input_img, output_img, kernels::kernel::SM_3x3_CF2_Vec);
	// call_kernel(input_img, output_img, kernels::kernel::SM_3x3_CF4_Vec);
	// call_kernel(input_img, output_img, kernels::kernel::SM_3x3_CF8_Vec);
	// call_kernel(input_img, output_img, kernels::kernel::SM_3x3_CF12_Vec);
	// call_kernel(input_img, output_img, kernels::kernel::SM_3x3_CF16_Vec);

	call_kernel(input_img, output_img, kernels::kernel::ArrayFire);
	call_kernel(input_img, output_img, kernels::kernel::OpenCV);
}

