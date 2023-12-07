#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>

#include "cuda_kernels.cuh"
#include "D_Tensor.cuh"
#include "Tensor.h"

D_Tensor::D_Tensor(int rows, int cols) : rows(rows), cols(cols), size(rows * cols), block_size(block_size) {
    gpuErrchk(cudaMalloc(&data, size*sizeof(float)));
    cudaDeviceSynchronize();
    block_size = 32;
}       

D_Tensor::~D_Tensor() {
    gpuErrchk(cudaFree(data));
    cudaDeviceSynchronize();
}

void D_Tensor::sequentialInit() {
    int grid_size = (size + block_size - 1) / block_size;
    d_sequentialInit<<<grid_size, block_size>>>(data, size);
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());
}

void D_Tensor::setOneInit(float val) {
    int grid_size = (size + block_size - 1) / block_size;
    d_setOneInit<<<grid_size, block_size>>>(data, size, val);
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());
}

Tensor* D_Tensor::to_cpu(){
    Tensor *mat = new Tensor(rows, cols);
    gpuErrchk(cudaMemcpy(mat->data, data, size*sizeof(float), cudaMemcpyDeviceToHost));

    return mat;
}

std::ostream& operator<<(std::ostream& os, const D_Tensor& D_Tensor) {
    float *temp_data = new float[D_Tensor.size];
    // gpuErrchk(cudaMemcpy(temp_data, data, D_Tensor.size*sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(temp_data, D_Tensor.data, D_Tensor.size*sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < D_Tensor.rows; ++i) {
        for (int j = 0; j < D_Tensor.cols; ++j) {
            os << temp_data[i * D_Tensor.cols + j] << ", ";
        }
        os << std::endl;
    }
    delete[] temp_data;
    return os;
}

D_Tensor* D_Tensor::operator*(const D_Tensor& other) const {
    std::cout << block_size << std::endl;
    // dim3 gridDim(ceil((rows+block_size)/block_size), ceil((block_size+rows)/block_size), 1);
    dim3 gridDim(int((rows+block_size)/block_size), int((other.cols+block_size)/block_size), 1);
    dim3 blockDim(block_size * block_size);

    // std::cout << gridDim.x  << ", " << gridDim.y << ", " << gridDim.z << std::endl;
    // std::cout << blockDim.x  << ", " << blockDim.y << ", " << blockDim.z << std::endl<< std::endl;

    D_Tensor *result = new D_Tensor(rows, other.cols);

    // sgemm_naive<<<gridDim, blockDim>>>(rows, cols, other.cols, data, other.data, result->data);
    mm_coalescing<<<gridDim, blockDim>>>(rows, cols, other.cols, block_size, data, other.data, result->data);
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    return result;
}


// D_Tensor::D_Tensor(const D_Tensor& other) : rows(other.rows), cols(other.cols), size(other.size) {
//     gpuErrchk(cudaMemcpy(other.data, data, size*sizeof(float), cudaMemcpyDeviceToDevice));
// }

void D_Tensor::transpose() {
    int temp;
    float *result;
    gpuErrchk(cudaMalloc(&result, size*sizeof(float)));

    int grid_size = (size + block_size - 1) / block_size;
    d_transpose<<<grid_size, block_size>>>(data, result, size, rows, cols);
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());
    
    gpuErrchk(cudaFree(data));
    data = result;

    temp = rows;
    rows = cols;
    cols = temp;
}

// // Need to change this to 1D
// D_Tensor** D_Tensor::vertical_split(int num_splits) {

//     int splitSize = cols / num_splits;
    
//     D_Tensor **result = new D_Tensor*[num_splits];
//     for (int i = 0; i < num_splits; ++i) {
//         result[i] = new D_Tensor(rows, splitSize);
//     }
    
//     for (int i = 0; i < num_splits; ++i) {
//         int col_offset = i * splitSize;
//         for (int row = 0; row < rows; ++row) {
//             for (int col = 0; col < splitSize; ++col) {
//                 result[i]->data[row * splitSize + col] = data[row * cols + col_offset + col];
//             }
//         }
//     }

//     return result;
// }