#include "cuda_kernels.cuh"

__global__ void d_setOneInit(float* data, int size, float val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = val;
    }
}

__global__ void d_sequentialInit(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = idx+1;
    }
}

__global__ void d_transpose(float *original, float *transposed, int size, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int x = idx / cols;
        int y = idx % cols;
        transposed[y * rows + x] = original[idx];
    }
}