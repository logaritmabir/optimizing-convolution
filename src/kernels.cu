#include "kernels.cuh"

#define BLOCKDIMX 16
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

template<int COARSENING_FACTOR>
__global__ void GM_3x3(imtype* __restrict__ input, imtype* __restrict__ output, const int rows, const int cols) {
	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * COARSENING_FACTOR;
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
		for (int i = 1; i < COARSENING_FACTOR; i++) {
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

template __global__ void GM_3x3<2>(imtype* __restrict__ input, imtype* __restrict__ output, const int rows, const int cols);
template __global__ void GM_3x3<4>(imtype* __restrict__ input, imtype* __restrict__ output, const int rows, const int cols);
template __global__ void GM_3x3<8>(imtype* __restrict__ input, imtype* __restrict__ output, const int rows, const int cols);
template __global__ void GM_3x3<12>(imtype* __restrict__ input, imtype* __restrict__ output, const int rows, const int cols);
template __global__ void GM_3x3<16>(imtype* __restrict__ input, imtype* __restrict__ output, const int rows, const int cols);

template<int COARSENING_FACTOR>
__global__ void CM_3x3(imtype* __restrict__ input, imtype* __restrict__ output, const int rows, const int cols) {
	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * COARSENING_FACTOR;
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
		+ CM_Filter[0][1] * frame[0][1]
		+ CM_Filter[0][2] * frame[0][2]
		+ CM_Filter[1][0] * frame[1][0]
		+ CM_Filter[1][1] * frame[1][1]
		+ CM_Filter[1][2] * frame[1][2]
		+ CM_Filter[2][0] * frame[2][0]
		+ CM_Filter[2][1] * frame[2][1]
		+ CM_Filter[2][2] * frame[2][2]);

		#pragma unroll
		for (int i = 1; i < COARSENING_FACTOR; i++) {
			int _ty = ty + i;
			shift_left(frame);
			if ((tx > 0 && tx < rows - 1) && (_ty > 0 && _ty < cols - 1)) {
				frame[0][2] = input[(tx - 1) * cols + _ty + 1];
				frame[1][2] = input[tx * cols + _ty + 1];
				frame[2][2] = input[(tx + 1) * cols + _ty + 1];

				output[(tx * cols + _ty)] = (GM_Filter[0][0] * frame[0][0]
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

template __global__ void CM_3x3<2>(imtype* __restrict__ input, imtype* __restrict__ output, const int rows, const int cols);
template __global__ void CM_3x3<4>(imtype* __restrict__ input, imtype* __restrict__ output, const int rows, const int cols);
template __global__ void CM_3x3<8>(imtype* __restrict__ input, imtype* __restrict__ output, const int rows, const int cols);
template __global__ void CM_3x3<12>(imtype* __restrict__ input, imtype* __restrict__ output, const int rows, const int cols);
template __global__ void CM_3x3<16>(imtype* __restrict__ input, imtype* __restrict__ output, const int rows, const int cols);

template<int COARSENING_FACTOR>
__global__ void GM_3x3_Vec(imtype* __restrict__ input, imtype* __restrict__ output, const int rows, const int cols)
{
	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * COARSENING_FACTOR;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	imtype vals[COARSENING_FACTOR] = { 0 };
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
		for (int i = 1; i < COARSENING_FACTOR; i++) {
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
		for (int i = 0; i < COARSENING_FACTOR; i += 4) {
			reinterpret_cast<imtype4*>(&output[(tx * cols + ty + i)])[0] = make_imtype4(vals[i], vals[i + 1], vals[i + 2], vals[i + 3]);
		}
	}
}

template __global__ void GM_3x3_Vec<2>(imtype* __restrict__ input, imtype* __restrict__ output, const int rows, const int cols);
template __global__ void GM_3x3_Vec<4>(imtype* __restrict__ input, imtype* __restrict__ output, const int rows, const int cols);
template __global__ void GM_3x3_Vec<8>(imtype* __restrict__ input, imtype* __restrict__ output, const int rows, const int cols);
template __global__ void GM_3x3_Vec<12>(imtype* __restrict__ input, imtype* __restrict__ output, const int rows, const int cols);
template __global__ void GM_3x3_Vec<16>(imtype* __restrict__ input, imtype* __restrict__ output, const int rows, const int cols);

template<int COARSENING_FACTOR>
__global__ void CM_3x3_Vec(imtype* __restrict__ input, imtype* __restrict__ output, const int rows, const int cols)
{
	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * COARSENING_FACTOR;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	imtype vals[COARSENING_FACTOR] = { 0 };
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
			+ CM_Filter[0][1] * frame[0][1]
			+ CM_Filter[0][2] * frame[0][2]
			+ CM_Filter[1][0] * frame[1][0]
			+ CM_Filter[1][1] * frame[1][1]
			+ CM_Filter[1][2] * frame[1][2]
			+ CM_Filter[2][0] * frame[2][0]
			+ CM_Filter[2][1] * frame[2][1]
			+ CM_Filter[2][2] * frame[2][2]);

		#pragma unroll
		for (int i = 1; i < COARSENING_FACTOR; i++) {
			int _ty = ty + i;
			shift_left(frame);
			if ((tx > 0 && tx < rows - 1) && (_ty > 0 && _ty < cols - 1)) {
				frame[0][2] = input[(tx - 1) * cols + _ty + 1];
				frame[1][2] = input[tx * cols + _ty + 1];
				frame[2][2] = input[(tx + 1) * cols + _ty + 1];

				vals[i] = (GM_Filter[0][0] * frame[0][0]
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
		for (int i = 0; i < COARSENING_FACTOR; i += 4) {
			reinterpret_cast<imtype4*>(&output[(tx * cols + ty + i)])[0] = make_imtype4(vals[i], vals[i + 1], vals[i + 2], vals[i + 3]);
		}
	}
}

template __global__ void CM_3x3_Vec<2>(imtype* __restrict__ input, imtype* __restrict__ output, const int rows, const int cols);
template __global__ void CM_3x3_Vec<4>(imtype* __restrict__ input, imtype* __restrict__ output, const int rows, const int cols);
template __global__ void CM_3x3_Vec<8>(imtype* __restrict__ input, imtype* __restrict__ output, const int rows, const int cols);
template __global__ void CM_3x3_Vec<12>(imtype* __restrict__ input, imtype* __restrict__ output, const int rows, const int cols);
template __global__ void CM_3x3_Vec<16>(imtype* __restrict__ input, imtype* __restrict__ output, const int rows, const int cols);


template<int COARSENING_FACTOR>
__global__ void SM_3x3(imtype* __restrict__ input, imtype* __restrict__ output, const int rows, const int cols)
{
	__shared__ imtype cache[BLOCKDIMY + 2][BLOCKDIMX * COARSENING_FACTOR + 2];

	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * COARSENING_FACTOR;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	int cy = threadIdx.x * COARSENING_FACTOR + 1;
	int cx = threadIdx.y + 1;

	if ((tx > 0 && tx < rows - 1) && (ty < cols - 1)) {
		#pragma unroll
		for (int i = 0; i < COARSENING_FACTOR; i++) {
			cache[cx][cy + i] = input[tx * cols + ty + i];
		}

		if (cx == 1) {
			#pragma unroll
			for (int i = 0; i < COARSENING_FACTOR; i++) {
				cache[0][cy + i] = input[(tx - 1) * cols + ty + i];
			}
		}
		if (cx == BLOCKDIMY) {
			#pragma unroll
			for (int i = 0; i < COARSENING_FACTOR; i++) {
				cache[BLOCKDIMY + 1][cy + i] = input[(tx + 1) * cols + ty + i];
			}
		}
		if (cy == 1) {
			cache[cx][0] = input[tx * cols + ty - 1];
		}
		if (cy == BLOCKDIMX * COARSENING_FACTOR - (COARSENING_FACTOR - 1)) {
			cache[cx][BLOCKDIMX * COARSENING_FACTOR + 1] = input[tx * cols + ty + COARSENING_FACTOR];
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
		for (int i = 1; i < COARSENING_FACTOR; i++) {
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

template<int COARSENING_FACTOR>
__global__ void SM_3x3_Vec(imtype* __restrict__ input, imtype* __restrict__ output, const int rows, const int cols)
{
	__shared__ imtype cache[BLOCKDIMY + 2][BLOCKDIMX * COARSENING_FACTOR + 2];

	int ty = (blockIdx.x * blockDim.x + threadIdx.x) * COARSENING_FACTOR;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	int cy = threadIdx.x * COARSENING_FACTOR + 1;
	int cx = threadIdx.y + 1;

	imtype4 vec4;
	imtype2 vec2;

	if ((tx > 0 && tx < rows - 1) && (ty < cols - 1)) {
		if (COARSENING_FACTOR == 2) {
			vec2 = reinterpret_cast<imtype2*>(&input[tx * cols + ty])[0];
			cache[cx][cy] = vec2.x;
			cache[cx][cy + 1] = vec2.y;
		} else {
			#pragma unroll
			for (int i = 0; i < COARSENING_FACTOR; i += 4) {
				vec4 = reinterpret_cast<imtype4*>(&input[tx * cols + ty + i])[0];
				cache[cx][cy + i] = vec4.x;
				cache[cx][cy + i + 1] = vec4.y;
				cache[cx][cy + i + 2] = vec4.z;
				cache[cx][cy + i + 3] = vec4.w;
			}
		}

		if (cx == 1) {
			if (COARSENING_FACTOR == 2) {
				vec2 = reinterpret_cast<imtype2*>(&input[(tx - 1) * cols + ty])[0];
				cache[0][cy] = vec2.x;
				cache[0][cy + 1] = vec2.y;
			} else {
				#pragma unroll
				for (int i = 0; i < COARSENING_FACTOR; i += 4) {
					vec4 = reinterpret_cast<imtype4*>(&input[(tx - 1) * cols + ty + i])[0];
					cache[0][cy + i] = vec4.x;
					cache[0][cy + i + 1] = vec4.y;
					cache[0][cy + i + 2] = vec4.z;
					cache[0][cy + i + 3] = vec4.w;
				}
			}
		}
		if (cx == BLOCKDIMY) {
			if (COARSENING_FACTOR == 2) {
				vec2 = reinterpret_cast<imtype2*>(&input[(tx + 1) * cols + ty])[0];
				cache[BLOCKDIMY + 1][cy] = vec2.x;
				cache[BLOCKDIMY + 1][cy + 1] = vec2.y;
			} else {
				#pragma unroll
				for (int i = 0; i < COARSENING_FACTOR; i += 4) {
					vec4 = reinterpret_cast<imtype4*>(&input[(tx + 1) * cols + ty + i])[0];
					cache[BLOCKDIMY + 1][cy + i] = vec4.x;
					cache[BLOCKDIMY + 1][cy + i + 1] = vec4.y;
					cache[BLOCKDIMY + 1][cy + i + 2] = vec4.z;
					cache[BLOCKDIMY + 1][cy + i + 3] = vec4.w;
				}
			}
		}
		if (cy == 1) {
			cache[cx][0] = input[tx * cols + ty - 1];
		}
		if (cy == BLOCKDIMX * COARSENING_FACTOR - (COARSENING_FACTOR - 1)) {
			cache[cx][BLOCKDIMX * COARSENING_FACTOR + 1] = input[tx * cols + ty + COARSENING_FACTOR];
		}
		__syncthreads();

		imtype vals[COARSENING_FACTOR] = { 0 };
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
		for (int i = 1; i < COARSENING_FACTOR; i++) {
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
		if (COARSENING_FACTOR == 2) {
			reinterpret_cast<imtype2*>(&output[(tx * cols + ty)])[0] = make_imtype2(vals[0], vals[1]);
		} else {
			#pragma unroll
			for (int i = 0; i < COARSENING_FACTOR; i += 4) {
				reinterpret_cast<imtype4*>(&output[(tx * cols + ty + i)])[0] = make_imtype4(vals[i], vals[i + 1], vals[i + 2], vals[i + 3]);
			}
		}
	}
}


#ifndef INSUFFICIENT_MEMORY_FOR_CF2
template __global__ void SM_3x3<2>(imtype* __restrict__ input, imtype* __restrict__ output, const int rows, const int cols);
template __global__ void SM_3x3_Vec<2>(imtype* __restrict__ input, imtype* __restrict__ output, const int rows, const int cols);
#endif

#ifndef INSUFFICIENT_MEMORY_FOR_CF4
template __global__ void SM_3x3<4>(imtype* __restrict__ input, imtype* __restrict__ output, const int rows, const int cols);
template __global__ void SM_3x3_Vec<4>(imtype* __restrict__ input, imtype* __restrict__ output, const int rows, const int cols);
#endif

#ifndef INSUFFICIENT_MEMORY_FOR_CF8
template __global__ void SM_3x3<8>(imtype* __restrict__ input, imtype* __restrict__ output, const int rows, const int cols);
template __global__ void SM_3x3_Vec<8>(imtype* __restrict__ input, imtype* __restrict__ output, const int rows, const int cols);
#endif

#ifndef INSUFFICIENT_MEMORY_FOR_CF12
template __global__ void SM_3x3<12>(imtype* __restrict__ input, imtype* __restrict__ output, const int rows, const int cols);
template __global__ void SM_3x3_Vec<12>(imtype* __restrict__ input, imtype* __restrict__ output, const int rows, const int cols);
#endif

#ifndef INSUFFICIENT_MEMORY_FOR_CF16
template __global__ void SM_3x3<16>(imtype* __restrict__ input, imtype* __restrict__ output, const int rows, const int cols);
template __global__ void SM_3x3_Vec<16>(imtype* __restrict__ input, imtype* __restrict__ output, const int rows, const int cols);
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

void testOutputs(cv::Mat* inputImg, cv::Mat* outputImg)
{
	imtype* d_input = nullptr;
	imtype* d_output = nullptr;
	imtype* h_input = reinterpret_cast<imtype*>(inputImg->data);
	imtype* h_output = reinterpret_cast<imtype*>(outputImg->data);

	const int cols = (*inputImg).cols;
	const int rows = (*inputImg).rows;
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
	GM_3x3<1> << <grid, block >> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	saveImage(h_output, rows, cols, "../images/outputs/GM_3x3.png");
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	CM_3x3<1> << <grid, block >> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	saveImage(h_output, rows, cols, "../images/outputs/CM_3x3.png");
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	SM_3x3<1> << <grid, block >> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	saveImage(h_output, rows, cols, "../images/outputs/SM_3x3.png");
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));
	#endif

	#if COARSENED_KERNELS
	GM_3x3<2> << <grid_cf2, block >> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	saveImage(h_output, rows, cols, "../images/outputs/GM_3x3_CF2.png");
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	GM_3x3<4> << <grid_cf4, block >> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	saveImage(h_output, rows, cols, "../images/outputs/GM_3x3_CF4.png");
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	GM_3x3<8> << <grid_cf8, block >> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	saveImage(h_output, rows, cols, "../images/outputs/GM_3x3_CF8.png");
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	GM_3x3<12> << <grid_cf12, block >> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	saveImage(h_output, rows, cols, "../images/outputs/GM_3x3_CF12.png");
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	GM_3x3<16> << <grid_cf16, block >> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	saveImage(h_output, rows, cols, "../images/outputs/GM_3x3_CF16.png");
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	CM_3x3<2> << <grid_cf2, block >> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	saveImage(h_output, rows, cols, "../images/outputs/CM_3x3_CF2.png");
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	CM_3x3<4> << <grid_cf4, block >> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	saveImage(h_output, rows, cols, "../images/outputs/CM_3x3_CF4.png");
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	CM_3x3<8> << <grid_cf8, block >> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	saveImage(h_output, rows, cols, "../images/outputs/CM_3x3_CF8.png");
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	CM_3x3<12> << <grid_cf12, block >> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	saveImage(h_output, rows, cols, "../images/outputs/CM_3x3_CF12.png");
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	CM_3x3<16> << <grid_cf16, block >> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	saveImage(h_output, rows, cols, "../images/outputs/CM_3x3_CF16.png");
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	#ifndef INSUFFICIENT_MEMORY_FOR_CF2
	SM_3x3<2> << <grid_cf2, block >> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	saveImage(h_output, rows, cols, "../images/outputs/SM_3x3_CF2.png");
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));
	#endif

	#ifndef INSUFFICIENT_MEMORY_FOR_CF4
	SM_3x3<4> << <grid_cf4, block >> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	saveImage(h_output, rows, cols, "../images/outputs/SM_3x3_CF4.png");
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));
	#endif

	#ifndef INSUFFICIENT_MEMORY_FOR_CF8
	SM_3x3<8> << <grid_cf8, block >> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	saveImage(h_output, rows, cols, "../images/outputs/SM_3x3_CF8.png");
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));
	#endif

	#ifndef INSUFFICIENT_MEMORY_FOR_CF12
	SM_3x3<12> << <grid_cf12, block >> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	saveImage(h_output, rows, cols, "../images/outputs/SM_3x3_CF12.png");
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));
	#endif

	#ifndef INSUFFICIENT_MEMORY_FOR_CF16
	SM_3x3<16> << <grid_cf16, block >> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	saveImage(h_output, rows, cols, "../images/outputs/SM_3x3_CF16.png");
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));
	#endif
	
	#endif

	#if VECTORIZED_KERNELS
	GM_3x3_Vec<2> << <grid_cf2, block>> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	saveImage(h_output, rows, cols, "../images/outputs/GM_3x3_CF2_Vec.png");
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	GM_3x3_Vec<4> << <grid_cf4, block>> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	saveImage(h_output, rows, cols, "../images/outputs/GM_3x3_CF4_Vec.png");
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	GM_3x3_Vec<8> << <grid_cf8, block>> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	saveImage(h_output, rows, cols, "../images/outputs/GM_3x3_CF8_Vec.png");
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	GM_3x3_Vec<12> << <grid_cf12, block>> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	saveImage(h_output, rows, cols, "../images/outputs/GM_3x3_CF12_Vec.png");
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	GM_3x3_Vec<16> << <grid_cf16, block>> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	saveImage(h_output, rows, cols, "../images/outputs/GM_3x3_CF16_Vec.png");
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	CM_3x3_Vec<2> << <grid_cf2, block>> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	saveImage(h_output, rows, cols, "../images/outputs/CM_3x3_CF2_Vec.png");
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	CM_3x3_Vec<4> << <grid_cf4, block>> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	saveImage(h_output, rows, cols, "../images/outputs/CM_3x3_CF4_Vec.png");
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	CM_3x3_Vec<8> << <grid_cf8, block>> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	saveImage(h_output, rows, cols, "../images/outputs/CM_3x3_CF8_Vec.png");
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	CM_3x3_Vec<12> << <grid_cf12, block>> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	saveImage(h_output, rows, cols, "../images/outputs/CM_3x3_CF12_Vec.png");
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	CM_3x3_Vec<16> << <grid_cf16, block>> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	saveImage(h_output, rows, cols, "../images/outputs/CM_3x3_CF16_Vec.png");
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));

	#ifndef INSUFFICIENT_MEMORY_FOR_CF2
	SM_3x3_Vec<2> << <grid_cf2, block>> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	saveImage(h_output, rows, cols, "../images/outputs/SM_3x3_CF2_Vec.png");
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));
	#endif

	#ifndef INSUFFICIENT_MEMORY_FOR_CF4
	SM_3x3_Vec<4> << <grid_cf4, block>> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	saveImage(h_output, rows, cols, "../images/outputs/SM_3x3_CF4_Vec.png");
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));
	#endif

	#ifndef INSUFFICIENT_MEMORY_FOR_CF8
	SM_3x3_Vec<8> << <grid_cf8, block>> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	saveImage(h_output, rows, cols, "../images/outputs/SM_3x3_CF8_Vec.png");
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));
	#endif

	#ifndef INSUFFICIENT_MEMORY_FOR_CF12
	SM_3x3_Vec<12> << <grid_cf12, block>> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	saveImage(h_output, rows, cols, "../images/outputs/SM_3x3_CF12_Vec.png");
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));
	#endif

	#ifndef INSUFFICIENT_MEMORY_FOR_CF16
	SM_3x3_Vec<16> << <grid_cf16, block>> > (d_input, d_output, rows, cols);
	CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
	saveImage(h_output, rows, cols, "../images/outputs/SM_3x3_CF16_Vec.png");
	CHECK_CUDA_ERROR(cudaMemset((void*)d_output, 0, size));
	#endif
	
	#endif

	#if NPP_KERNEL && defined(IMTYPE_FLOAT)
	{
		CHECK_NPP_ERROR(nppiFilterGauss_32f_C1R(d_input, inputImg->step, d_output, inputImg->step, {cols, rows}, NPP_MASK_SIZE_3_X_3));
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
			d_input.upload(*inputImg);
			d_output.upload(*outputImg);

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

using kernelPtr = void (*)(imtype* __restrict__ input, imtype* __restrict__ output, const int rows, const int cols);

std::map<KernelType, kernelPtr> kernelMap = {
	{KernelType::GM_3x3, &GM_3x3<1>},
	{KernelType::CM_3x3, &CM_3x3<1>},
	{KernelType::SM_3x3, &SM_3x3<1>},
	{KernelType::GM_3x3_CF2, &GM_3x3<2>},
	{KernelType::GM_3x3_CF4, &GM_3x3<4>},
	{KernelType::GM_3x3_CF8, &GM_3x3<8>},
	{KernelType::GM_3x3_CF12, &GM_3x3<12>},
	{KernelType::GM_3x3_CF16, &GM_3x3<16>},
	{KernelType::CM_3x3_CF2, &CM_3x3<2>},
	{KernelType::CM_3x3_CF4, &CM_3x3<4>},
	{KernelType::CM_3x3_CF8, &CM_3x3<8>},
	{KernelType::CM_3x3_CF12, &CM_3x3<12>},
	{KernelType::CM_3x3_CF16, &CM_3x3<16>},
	{KernelType::GM_3x3_CF2_Vec, &GM_3x3_Vec<2>},
	{KernelType::GM_3x3_CF4_Vec, &GM_3x3_Vec<4>},
	{KernelType::GM_3x3_CF8_Vec, &GM_3x3_Vec<8>},
	{KernelType::GM_3x3_CF12_Vec, &GM_3x3_Vec<12>},
	{KernelType::GM_3x3_CF16_Vec, &GM_3x3_Vec<16>},
	{KernelType::CM_3x3_CF2_Vec, &CM_3x3_Vec<2>},
	{KernelType::CM_3x3_CF4_Vec, &CM_3x3_Vec<4>},
	{KernelType::CM_3x3_CF8_Vec, &CM_3x3_Vec<8>},
	{KernelType::CM_3x3_CF12_Vec, &CM_3x3_Vec<12>},
	{KernelType::CM_3x3_CF16_Vec, &CM_3x3_Vec<16>},
	{KernelType::SM_3x3_CF2, &SM_3x3<2>},
	{KernelType::SM_3x3_CF4, &SM_3x3<4>},
	{KernelType::SM_3x3_CF8, &SM_3x3<8>},
	{KernelType::SM_3x3_CF12, &SM_3x3<12>},
	{KernelType::SM_3x3_CF16, &SM_3x3<16>},
	{KernelType::SM_3x3_CF2_Vec, &SM_3x3_Vec<2>},
	{KernelType::SM_3x3_CF4_Vec, &SM_3x3_Vec<4>},
	{KernelType::SM_3x3_CF8_Vec, &SM_3x3_Vec<8>},
	{KernelType::SM_3x3_CF12_Vec, &SM_3x3_Vec<12>},
	{KernelType::SM_3x3_CF16_Vec, &SM_3x3_Vec<16>}
};

void callKernel(cv::Mat* inputImg, cv::Mat* outputImg, KernelType kernelType){
	imtype* d_input = nullptr;
	imtype* d_output = nullptr;
	imtype* h_input = reinterpret_cast<imtype*>(inputImg->data);
	imtype* h_output = reinterpret_cast<imtype*>(outputImg->data);

	const int cols = (*inputImg).cols;
	const int rows = (*inputImg).rows;
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

	auto kernelFunc = kernelMap.find(kernelType);
	if (kernelFunc != kernelMap.end()) {
		switch (kernelType) {
			case KernelType::GM_3x3_CF2:
			case KernelType::CM_3x3_CF2:
			case KernelType::GM_3x3_CF2_Vec:
			case KernelType::CM_3x3_CF2_Vec:
			case KernelType::SM_3x3_CF2:
			case KernelType::SM_3x3_CF2_Vec:
				kernelFunc->second<<<grid_cf2, block>>>(d_input, d_output, rows, cols);
				break;
			case KernelType::GM_3x3_CF4:
			case KernelType::CM_3x3_CF4:
			case KernelType::GM_3x3_CF4_Vec:
			case KernelType::CM_3x3_CF4_Vec:
			case KernelType::SM_3x3_CF4:
			case KernelType::SM_3x3_CF4_Vec:
				kernelFunc->second<<<grid_cf4, block>>>(d_input, d_output, rows, cols);
				break;
			case KernelType::GM_3x3_CF8:
			case KernelType::CM_3x3_CF8:
			case KernelType::GM_3x3_CF8_Vec:
			case KernelType::CM_3x3_CF8_Vec:
			case KernelType::SM_3x3_CF8:
			case KernelType::SM_3x3_CF8_Vec:
				kernelFunc->second<<<grid_cf8, block>>>(d_input, d_output, rows, cols);
				break;
			case KernelType::GM_3x3_CF12:
			case KernelType::CM_3x3_CF12:
			case KernelType::GM_3x3_CF12_Vec:
			case KernelType::CM_3x3_CF12_Vec:
			case KernelType::SM_3x3_CF12:
			case KernelType::SM_3x3_CF12_Vec:
				kernelFunc->second<<<grid_cf12, block>>>(d_input, d_output, rows, cols);
				break;
			case KernelType::GM_3x3_CF16:
			case KernelType::CM_3x3_CF16:
			case KernelType::GM_3x3_CF16_Vec:
			case KernelType::CM_3x3_CF16_Vec:
			case KernelType::SM_3x3_CF16:
			case KernelType::SM_3x3_CF16_Vec:
				kernelFunc->second<<<grid_cf16, block>>>(d_input, d_output, rows, cols);
				break;
			case KernelType::ArrayFire:
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
					CHECK_ARRAYFIRE(af_create_array(&af_input, h_input, 2, dims_of_input, u8));	
					#endif
					CHECK_ARRAYFIRE(af_gaussian_kernel(&af_kernel, 3, 3, 0, 0));
					CHECK_ARRAYFIRE(af_convolve2(&af_output, af_input, af_kernel, AF_CONV_DEFAULT, AF_CONV_AUTO));
					CHECK_ARRAYFIRE(af_get_data_ptr(h_output, af_output));
				}
				break;
			case KernelType::OpenCV:
				{
					cv::cuda::GpuMat d_input, d_output;
					d_input.upload(*inputImg);
					d_output.upload(*outputImg);

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
				break;
			default:
				kernelFunc->second<<<grid, block>>>(d_input, d_output, rows, cols);
				break;
		}
	} else {
		std::cerr << "Kernel type not supported." << std::endl;
	}

	CHECK_CUDA_ERROR(cudaHostUnregister(h_input));
	CHECK_CUDA_ERROR(cudaHostUnregister(h_output));
	CHECK_CUDA_ERROR(cudaFree(d_input));
	CHECK_CUDA_ERROR(cudaFree(d_output));
	CHECK_CUDA_ERROR(cudaDeviceReset());
}


void launchKernels(cv::Mat* inputImg, cv::Mat* outputImg){
	callKernel(inputImg, outputImg, KernelType::GM_3x3);
	callKernel(inputImg, outputImg, KernelType::GM_3x3_CF2);
	callKernel(inputImg, outputImg, KernelType::GM_3x3_CF4);
	callKernel(inputImg, outputImg, KernelType::GM_3x3_CF8);
	callKernel(inputImg, outputImg, KernelType::GM_3x3_CF12);
	callKernel(inputImg, outputImg, KernelType::GM_3x3_CF16);
	callKernel(inputImg, outputImg, KernelType::GM_3x3_CF2_Vec);
	callKernel(inputImg, outputImg, KernelType::GM_3x3_CF4_Vec);
	callKernel(inputImg, outputImg, KernelType::GM_3x3_CF8_Vec);
	callKernel(inputImg, outputImg, KernelType::GM_3x3_CF12_Vec);
	callKernel(inputImg, outputImg, KernelType::GM_3x3_CF16_Vec);

	callKernel(inputImg, outputImg, KernelType::CM_3x3);
	callKernel(inputImg, outputImg, KernelType::CM_3x3_CF2);
	callKernel(inputImg, outputImg, KernelType::CM_3x3_CF4);
	callKernel(inputImg, outputImg, KernelType::CM_3x3_CF8);
	callKernel(inputImg, outputImg, KernelType::CM_3x3_CF12);
	callKernel(inputImg, outputImg, KernelType::CM_3x3_CF16);
	callKernel(inputImg, outputImg, KernelType::CM_3x3_CF2_Vec);
	callKernel(inputImg, outputImg, KernelType::CM_3x3_CF4_Vec);
	callKernel(inputImg, outputImg, KernelType::CM_3x3_CF8_Vec);
	callKernel(inputImg, outputImg, KernelType::CM_3x3_CF12_Vec);
	callKernel(inputImg, outputImg, KernelType::CM_3x3_CF16_Vec);

	callKernel(inputImg, outputImg, KernelType::SM_3x3);
	callKernel(inputImg, outputImg, KernelType::SM_3x3_CF2);
	callKernel(inputImg, outputImg, KernelType::SM_3x3_CF4);
	callKernel(inputImg, outputImg, KernelType::SM_3x3_CF8);
	callKernel(inputImg, outputImg, KernelType::SM_3x3_CF12);
	callKernel(inputImg, outputImg, KernelType::SM_3x3_CF16);
	callKernel(inputImg, outputImg, KernelType::SM_3x3_CF2_Vec);
	callKernel(inputImg, outputImg, KernelType::SM_3x3_CF4_Vec);
	callKernel(inputImg, outputImg, KernelType::SM_3x3_CF8_Vec);
	callKernel(inputImg, outputImg, KernelType::SM_3x3_CF12_Vec);
	callKernel(inputImg, outputImg, KernelType::SM_3x3_CF16_Vec);

	// callKernel(inputImg, outputImg, KernelType::ArrayFire);
	// callKernel(inputImg, outputImg, KernelType::OpenCV);
}