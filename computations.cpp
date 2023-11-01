#include <iostream>
#include "exceptions.h"

float** matmul(float **input, int inp_rows, int inp_cols, float **weights, int w_rows, int w_cols){
    if (inp_cols != w_cols){
        throw SizeMismatchException(inp_rows, inp_cols, w_cols, w_rows);
    }

    float **result;

    return result;
}