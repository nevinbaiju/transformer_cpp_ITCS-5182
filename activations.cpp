#include <iostream>
#include <vector>
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