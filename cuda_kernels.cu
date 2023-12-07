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

__global__ void sgemm_naive(int M, int N, int K,
                            const float *A, const float *B, float *C) {
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (x < M && y < K) {
    float tmp = 0.0;
    for (int i = 0; i < N; ++i) {
      tmp += A[y * N + i] * B[i * K + x];
    }
    C[y * N + x] = tmp;
  }
}
