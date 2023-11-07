#include <iostream>
#include <vector>
#include "activations.h"
#include "initializations.h"
#include "helpers.h"
#include "computations.h"

int main(int argc, char *argv[]) {
    int size = 5;
    int inp_row = atoi(argv[1]), inp_col=atoi(argv[2]);
    int w_row = inp_col, w_col=atoi(argv[3]);

    float *weight = new float[w_row*w_col];
    float *input = new float[inp_row*inp_col];
    
    setOneInit(input, inp_row, inp_col, -1);
    print_arr(input, inp_row, inp_col);
    setOneInit(weight, w_row, w_col, 3);
    print_arr(weight, w_row, w_col);
    
    float *result;
    result = matmul(input, inp_row, inp_col, weight, w_row, w_col);
    result = softmax(result, inp_row, w_col, false);
    print_arr(result, inp_row, w_col);
    

    return 0;
}
