#include <iostream>
#include <vector>
#include <chrono>

#include "activations.h"
#include "initializations.h"
#include "helpers.h"
#include "computations.h"
#include "attention.h"

int main(int argc, char *argv[]) {
    int rows = atoi(argv[1]), cols = atoi(argv[2]);
    int embedding = cols;
    int num_heads = atoi(argv[3]);

    float *arr1 = new float[rows*cols];
    float *arr2 = new float[rows*cols];
    float *arr3 = new float[rows*cols];

    sequentialInit(arr1, rows, cols);
    sequentialInit(arr2, rows, cols);
    sequentialInit(arr3, rows, cols);

    auto start = std::chrono::high_resolution_clock::now();
    float *result = multi_head_attention(arr1, rows, cols,
                                        arr2, rows, cols,
                                        arr3, rows, cols, 
                                        4, rows, false);
    auto end = std::chrono::high_resolution_clock::now();                                        

    get_bench_results(rows, cols, embedding, num_heads, start, end);  

    delete[] arr1;
    delete[] arr2;
    delete[] arr3;
    delete[] result;                                 
}
