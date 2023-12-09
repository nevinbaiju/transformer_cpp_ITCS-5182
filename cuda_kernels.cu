#include "cuda_kernels.cuh"
#include <stdio.h>
#include <algorithm>

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

__global__ void d_scale(float* data, int size, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = data[idx]/scale;
    }
}

__global__ void _d_softmax(float *input, int rows, int cols){
    int row_id = blockIdx.x * blockDim.x + threadIdx.x;
    int start_index = row_id*cols;
    int end_index = start_index+cols;

    float max_val=input[start_index], sum_val=0;

    for(int i=start_index+1; i<end_index; i++){
        if (max_val < input[i]){
            max_val = input[i];
        }
    }
    
    for(int i=start_index; i<end_index; i++){
        input[i] = exp(input[i] - max_val);
        sum_val += input[i];
    }
    
    for(int i=start_index; i<end_index; i++){
        input[i] = input[i]/sum_val;
    }
}

__global__ void sgemm_naive(int M, int K, int N, 
                            const float *A, const float *B, float *C) {
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < M && y < N) {
        float tmp = 0.0;
        for (int i = 0; i < K; ++i) {
            tmp += A[x * K + i] * B[i * N + y];
        }
        C[x * N + y] = tmp;
    }
}

__global__ void mm_coalescing(int M, int K, int N, int block_size,
                            const float *A, const float *B, float *C) {                                
    const int x = blockIdx.x * block_size + (threadIdx.x / block_size);
    const int y = blockIdx.y * block_size + (threadIdx.x % block_size);

    if (x < M && y < N) {
        float tmp = 0.0;
        for (int i = 0; i < K; ++i) {
            tmp += A[x * K + i] * B[i * N + y];
        }
        // printf("%d, %d: %f\n", x, y, tmp);
        // printf("%d, %d, %d\n", blockIdx.y, blockDim.x, threadIdx.x);
        C[x * N + y] = tmp;
    }
}