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

void setOneInit(float* array, int rows, int cols, int val) {
    std::cout << "Setting val: " << val << std::endl;
    for (int i = 0; i <rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            array[i*cols+j] = val;
        }
    }
}

float *transpose(float* array, int rows, int cols, bool inplace) {
    float temp, *result;
    if (inplace){
        result = array;
    }
    else{
        result = new float[rows*cols];
    }
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << i << ":" << j << "..." << array[i*cols + j] << " swapped with" << array[j*rows + i] << std::endl;
            temp = array[i * cols + j];
            result[i * cols + j] = array[j * cols + i];
            result[j * rows + i] = temp;
        }
    }
    return result;
}