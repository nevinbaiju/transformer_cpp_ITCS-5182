#include <iostream>
#include <math.h>
#include "activations.h"

float* relu(float input[], int size, bool inplace) {
    float* result;
    if(!inplace){
        result = new float[size];
    }
    for (int i = 0; i < size; ++i) {
        if (inplace) {
            input[i] = std::max(input[i], 0.0f);
        } else {
            result[i] = std::max(input[i], 0.0f);
        }
    }

    if (inplace) {
        return input;  // Return the input array reference
    } else {
        return result; // Return the new result array reference
    }
}

float* relu(float *input, int rows, int cols, bool inplace) {
    float* result;
    int size = rows*cols;
    if(inplace){
        result = input;
    }
    else{
        result = new float[size];
    }

    for (int i = 0; i < size; ++i) {
        result[i] = std::max(input[i], 0.0f);
    }

    return result;
}

float* softmax(float *input, int rows, int cols, bool inplace){
    float sum_val;
    float *result;

    if (inplace){
        result = input;
    }
    else{
        result = new float[rows*cols];
    }
    int i, j;
    for(i=0; i<rows; i++){
        sum_val = 0;
        for(j=0; j<cols; j++){
            result[i*cols + j] = exp(input[i*cols + j]);
            sum_val += result[i*cols + j];
        }
        for(j=0; j<cols; j++){
            result[i*cols + j] = result[i*cols + j]/sum_val;
        }
    }

    return result;
}