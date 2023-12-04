#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>

#include "cuda_kernels.cuh"
#include "D_Tensor.cuh"
#include "Tensor.h"

D_Tensor::D_Tensor(int rows, int cols) : rows(rows), cols(cols), size(rows * cols), block_size(block_size) {
    gpuErrchk(cudaMalloc(&data, size*sizeof(float)));
}       

D_Tensor::~D_Tensor() {
    gpuErrchk(cudaFree(data));
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

// D_Tensor* D_Tensor::operator*(const D_Tensor& other) const {

//     // std::cout << "Mult" << std::endl;
//     // std::cout << *this << " \n With \n" << other << std::endl;
//     if (cols != other.rows) {
//         throw SizeMismatchException(rows, cols, other.rows, other.cols);
//     }
//     D_Tensor *result = new D_Tensor(rows, other.cols);
//     result->setOneInit(0);

//     #ifdef NORMAL
//         for (int i = 0; i < rows; i++) {
//             for (int j = 0; j < other.cols; j++) {
//                 result->data[i * other.cols + j] = 0;
//                 for (int k = 0; k < cols; k++) {
//                     result->data[i * other.cols + j] += data[i * cols + k] * other.data[k * other.cols + j];
//                 }
//             }
//         }
//     #else
//         #ifdef AVX
//             int i, j, k, pipeline_id;
//             const int num_pipeline = 4;
//             __m256 a[num_pipeline], b[num_pipeline], result_register;
//             float temp_result[8];

//             #pragma omp parallel for private(a, b, result_register, temp_result, i, j, k, pipeline_id)
//             for(i=0; i<rows; i+=8){
//                 float temp_result[8];
//                 j = 0;
//                 for(j=0; j<cols; j+=num_pipeline){
//                     for (k=0; k<other.cols; k++){
//                         result_register = _mm256_set_ps(result->data[(i)*other.cols + k], result->data[(i+1)*other.cols + k], 
//                                                         result->data[(i+2)*other.cols + k], result->data[(i+3)*other.cols + k], 
//                                                         result->data[(i+4)*other.cols + k], result->data[(i+5)*other.cols + k], 
//                                                         result->data[(i+6)*other.cols + k], result->data[(i+7)*other.cols + k]);
//                         #pragma unroll
//                         for(pipeline_id = 0; pipeline_id < num_pipeline; pipeline_id++){
//                             a[pipeline_id] = _mm256_set_ps(data[(i)*cols + j+pipeline_id], data[(i+1)*cols + j+pipeline_id], 
//                                                            data[(i+2)*cols + j+pipeline_id], data[(i+3)*cols + j+pipeline_id], 
//                                                            data[(i+4)*cols + j+pipeline_id], data[(i+5)*cols + j+pipeline_id], 
//                                                            data[(i+6)*cols + j+pipeline_id], data[(i+7)*cols + j+pipeline_id]);
//                             b[pipeline_id] = _mm256_set1_ps(other.data[(j+pipeline_id)*other.cols + k]);
//                             result_register = _mm256_fmadd_ps(a[pipeline_id], b[pipeline_id], result_register);
//                         }                                                        
                        
//                         _mm256_storeu_ps(&temp_result[0], result_register);

//                         #pragma unroll
//                         for(int res_id=0; res_id<8; res_id++){
//                             result->data[(i+res_id)*other.cols + k] = temp_result[res_id];
//                         }
//                     }
//                 }
//             }
//         #endif
//     #endif
    
//     // std::cout << "Result: " << result << "\n end mult" << std::endl;

//     return result;
// }


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