#include "d_activations.cuh"
#include "cuda_kernels.cuh"
#include "D_Tensor.cuh"

#include <math.h>
#include <cstring>

D_Tensor* scale(D_Tensor *mat, bool inplace) {
    D_Tensor *result;

    if (inplace) {
        result = mat;
    } else {
        result = new D_Tensor(mat->rows, mat->cols);
        std::memcpy(result->data, mat->data, sizeof(float) * mat->size);
    }

    float scale_val = std::sqrt(result->cols);

    int grid_size = (result->size + result->block_size - 1) / result->block_size;
    d_scale<<<grid_size, result->block_size>>>(result->data, result->size, scale_val);
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());
    
    return result;
}

D_Tensor* softmax(D_Tensor *mat, bool inplace){
    D_Tensor *result;

    if (inplace){
        result = mat;
    }
    else{
        result = new D_Tensor(mat->rows, mat->cols);
        std::memcpy(result->data, mat->data, sizeof(float) * mat->size);
    }

    int grid_size = (result->rows + result->block_size - 1) / result->block_size;
    _d_softmax<<<grid_size, result->block_size>>>(result->data, result->rows, result->cols);
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    return result;
}