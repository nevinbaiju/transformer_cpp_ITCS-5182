#include <iostream>
#include <vector>
#include "activations.h"
#include "initializations.h"
#include "helpers.h"
#include "computations.h"
#include "attention.h"

int main(int argc, char *argv[]) {
    int rows = 100, cols = 1000000;

    float *arr1 = new float[rows*cols];
    float *arr2 = new float[rows*cols];
    float *arr3 = new float[rows*cols];

    sequentialInit(arr1, rows, cols);
    sequentialInit(arr2, rows, cols);
    sequentialInit(arr3, rows, cols);

    float *result = multi_head_attention(arr1, rows, cols,
                                        arr2, rows, cols,
                                        arr3, rows, cols, 
                                        4, rows, false);

    // print_arr(result, rows, cols);                                        
}
