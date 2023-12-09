#include <cuda_runtime.h>

__global__ void d_setOneInit(float* data, int size, float val);
__global__ void d_sequentialInit(float* data, int size);
__global__ void d_transpose(float *original, float *transposed, int size, int rows, int cols);

__global__ void d_scale(float* data, int size, float scale);
__global__ void _d_softmax(float *input, int rows, int cols);

// Matmul kernels
__global__ void sgemm_naive(int M, int N, int K,
                            const float *A, 
                            const float *B, float *C);
__global__ void mm_coalescing(int M, int K, int N, int block_size,
                             const float *A, 
                             const float *B, float *C);

