#include <iostream>
#include <vector>
#include "activations.h"
#include "initializations.h"
#include "helpers.h"
#include "computations.h"

int main() {
    int size = 5;
    int inp_row = 5, inp_col=5;
    int w_row = inp_col, w_col=1;

    float *weight = new float[w_row*w_col];
    float *input = new float[inp_row*inp_col];
    
    setOneInit(weight, w_row, w_col);
    setOneInit(input, inp_row, inp_col);

    print_arr(weight, w_row, w_col);
    print_arr(input, inp_row, inp_col);
    
    float *result;
    result = matmul(input, inp_row, inp_col, weight, w_row, w_col);
    print_arr(result, inp_row, w_col);
    

    return 0;
}
