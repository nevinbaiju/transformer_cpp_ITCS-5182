#include <random>
#include <cmath>
#include <iostream>
#include <vector>
#include <stdio.h>

#include "initializations.h"
#include "Tensor.h"
#ifdef CUDA
#include "D_Tensor.cuh"
#include <cuda_runtime.h>
#endif

void kaimingInit(float** array, int rows, int cols, int fan_in) {
    std::random_device rd;
    std::mt19937 gen(rd());

    const float bound = sqrt(2.0 / fan_in);
    std::normal_distribution<float> distribution(0.0f, bound);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            array[i][j] = distribution(gen);
        }
    }
}

void identityInit(float *array, int rows, int cols){
    for(int i=0; i<rows; i++){
        array[i*cols + i] = 1;
    }
}

void sequentialInit(float *array, int rows, int cols){
    for(int i=0; i<rows*cols; i++){
        array[i] = i+1;
    }
}

void setOneInit(float* array, int rows, int cols, int val) {
    for (int i = 0; i <rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            array[i*cols+j] = val;
        }
    }
}

float *transpose(float* array, int rows, int cols, bool inplace) {
    float temp;
    float *result = new float[rows*cols];
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[j*rows + i] = array[i*cols + j];
        }
    }
    if (inplace){
        for(int i=0; i<rows*cols; i++){
            array[i] = result[i];
        }
        return array;
    }
    else{
        return result;
    }
}

// Need to change this to 1D
float** vertical_split(float *matrix, int rows, int cols, int num_splits) {

    int splitSize = cols / num_splits;

    // Allocate memory for the result array
    float **result = new float*[num_splits];
    for (int i = 0; i < num_splits; ++i) {
        result[i] = new float[rows * splitSize];
    }

    // Perform vertical splits
    for (int i = 0; i < num_splits; ++i) {
        int col_offset = i * splitSize;
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < splitSize; ++col) {
                result[i][row * splitSize + col] = matrix[row * cols + col_offset + col];
            }
        }
    }

    return result;
}

Tensor *vertical_concat(std::vector<Tensor*> tensors){
    int tens_len = tensors.size();
    int tens_cols = tensors[0]->cols;
    int tens_rows = tensors[0]->rows;

    Tensor *result = new Tensor(tens_rows, tens_cols*tens_len);
    int tens_index, tens_col_index;
    for (int i=0; i<tens_rows; i++){
        for (int j=0; j<result->cols; j++){
            tens_index = j/tens_cols;
            tens_col_index = j%tens_cols;

            result->data[i*result->cols + j] = tensors[tens_index]->data[i*tens_cols + tens_col_index];
        }
    }

    return result;
}

#ifdef CUDA
D_Tensor *vertical_concat(std::vector<D_Tensor*> tensors){
    int tens_len = tensors.size();
    int tens_cols = tensors[0]->cols;
    int tens_rows = tensors[0]->rows;

    D_Tensor *result = new D_Tensor(tens_rows, tens_cols*tens_len);
    int tens_col_index, result_cols = tens_cols*tens_len;
    for (int i=0; i<tens_rows; i++){
        for (int tens_id=0; tens_id<tens_len; tens_id++){
            printf("row: %d, tens_id: %d\n", i, tens_id);
            tens_col_index = tens_cols * tens_id;
            // gpuErrchk(cudaMemcpy(result->data, tensors[0]->data, 3 * sizeof(float), cudaMemcpyDeviceToDevice));
            gpuErrchk(cudaMemcpy(result->data + (result_cols*i) + tens_id*tens_cols, 
                                 tensors[tens_id]->data + (i*tens_cols), 
                                 tens_cols * sizeof(float), cudaMemcpyDeviceToDevice));
        }
    }
    return result;
}
#endif