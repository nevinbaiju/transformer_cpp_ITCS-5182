#include <iostream>
#include <vector>
#include "activations.h"
#include "initializations.h"
#include "helpers.h"
#include "computations.h"
#include "attention.h"

int main(int argc, char *argv[]) {
    int rows = 4, cols = 4;

    float *input_arr1 = new float[rows*cols];
    float *input_arr2 = new float[rows*cols];
    float *input_arr3 = new float[rows*cols];
    sequentialInit(input_arr1, rows, cols);
    sequentialInit(input_arr2, rows, cols);
    sequentialInit(input_arr3, rows, cols);

    float *result = dot_product_attention(input_arr1, rows, cols,
                                          input_arr2, rows, cols,
                                          input_arr3, rows, cols);    
    print_arr(result, rows, cols);                                                
}
