#include <iostream>
#include <vector>
#include "activations.h"
#include "initializations.h"
#include "helpers.h"
#include "computations.h"
#include "attention.h"

int main(int argc, char *argv[]) {
    int rows = 2, cols = 4;

    float *input_arr1 = new float[rows*cols];
    float *input_arr2 = new float[rows*cols];
    float *input_arr3 = new float[rows*cols];
    sequentialInit(input_arr1, rows, cols);
    sequentialInit(input_arr2, rows, cols);
    sequentialInit(input_arr3, rows, cols);
    
    print_arr(input_arr1, rows, cols);
    float **result = vertical_split(input_arr1, rows, cols, 2);  
    print_arr(result[0], rows, 2);      
    print_arr(result[1], rows, 2);                                                
}
