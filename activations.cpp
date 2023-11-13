#include <iostream>
#include <math.h>
#include <cstring>
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

void _softmax(float *input, int start_index, int end_index){
    float max_val=input[start_index], sum_val=0;
    for(int i=start_index+1; i<end_index; i++){
        max_val = std::max(max_val, input[i]);
    }
    for(int i=start_index; i<end_index; i++){
        input[i] = exp(input[i] - max_val);
        sum_val += input[i];
    }
    for(int i=start_index; i<end_index; i++){
        input[i] = input[i]/sum_val;
    }
}

float* softmax(float *input, int rows, int cols, bool inplace){
    float sum_val;
    float *result;

    if (inplace){
        result = input;
    }
    else{
        result = new float[rows*cols];
        std::memcpy(result, input, rows*cols*sizeof *input);
    }
    int i, j;
    for(i=0; i<rows; i++){
        _softmax(result, i*cols, (i+1)*cols);
    }

    return result;
}

float* scale(float *input, int rows, int cols, bool inplace){
    float *result;

    if (!inplace){
        result = new float[rows*cols];
    }
    else{
        result = input;
    }
    float scale = sqrt(cols);
    for(int i=0; i<rows*cols; i++){
        result[i] = result[i]/scale;
    }

    return result;
}