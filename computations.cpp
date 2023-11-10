#include <iostream>
#include "exceptions.h"
#include "initializations.h"
#include "helpers.h"

float* matmul(float *input, int inp_rows, int inp_cols, float *weights, int w_rows, int w_cols){
    if (inp_cols != w_rows){
        if (inp_cols != w_cols){
            throw SizeMismatchException(inp_rows, inp_cols, w_rows, w_cols);
        }
        transpose(weights, w_rows, w_cols, true);
        int temp = w_rows;
        w_rows = w_cols;
        w_cols = temp;
    }

    float *result = new float[inp_rows*w_cols]; 

    for (int i = 0; i < inp_rows; i++) {
        for (int j = 0; j < w_cols; j++) {
            result[i*w_cols + j] = 0.0;
            for (int k = 0; k < inp_cols; k++) {
                result[i*w_cols + j] += input[i*inp_cols + k] * weights[k*w_cols + j];
            }
        }
    }

    return result;
}