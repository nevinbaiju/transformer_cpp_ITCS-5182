#include <iostream>
#include "exceptions.h"

float* matmul(float *input, int inp_rows, int inp_cols, float *weights, int w_rows, int w_cols){
    if (inp_cols != w_rows){
        throw SizeMismatchException(inp_rows, inp_cols, w_cols, w_rows);
    }

    float *result = new float[inp_rows*w_cols]; 

    for (int i = 0; i < inp_rows; i++) {
        for (int j = 0; j < w_cols; j++) {
            result[i*inp_rows + j] = 0.0;
            for (int k = 0; k < inp_cols; k++) {
                result[i*inp_rows + j] += input[i*inp_rows + k] * weights[k*w_rows + j];
            }
        }
    }

    return result;
}