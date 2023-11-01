#include <iostream>
#include "exceptions.h"

float** matmul(float **input, int inp_rows, int inp_cols, float **weights, int w_rows, int w_cols){
    if (inp_cols != w_rows){
        throw SizeMismatchException(inp_rows, inp_cols, w_cols, w_rows);
    }

    float **result = new float*[inp_rows]; 
    for(int i=0; i<inp_rows; i++){
        result[i] = new float[w_cols];
    }

    for (int i = 0; i < inp_rows; i++) {
        for (int j = 0; j < w_cols; j++) {
            result[i][j] = 0.0;
            for (int k = 0; k < inp_cols; k++) {
                result[i][j] += input[i][k] * weights[k][j];
            }
        }
    }

    return result;
}