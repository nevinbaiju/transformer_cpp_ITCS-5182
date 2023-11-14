#include <random>
#include <cmath>
#include "initializations.h"
#include <iostream>

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