#include <iostream>
#include <iomanip>
#include <cstring>
#include <chrono>

const double cpu_peak_gflops = 1881e9;

void print_arr(float arr[], int arr_size){
    for(int i=0; i<arr_size; i++){
        std::cout << arr[i] << ", ";
    }
    std::cout << "\n";
}

void print_arr(float *arr, int rows, int cols){
    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){
            std::cout << std::setw(5) << std::setfill(' ') << arr[i*cols + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "\n";
}

void get_bench_results(int rows, int cols, int embedding, int num_heads, 
                         std::chrono::time_point<std::chrono::high_resolution_clock> start, 
                         std::chrono::time_point<std::chrono::high_resolution_clock> end){
    u_int64_t flop = 6*rows*cols*embedding + num_heads*(2*(cols/num_heads)*embedding*(cols/num_heads)) + 
                      2*num_heads*(2*(cols/num_heads)*(cols/num_heads)) + embedding*num_heads*(2*(cols/num_heads)*(cols/num_heads));
    std::chrono::duration<double> elapsed_seconds = end-start;
    double seconds = elapsed_seconds.count();
    double flops = flop/(seconds*1e9);

    double theoretical_elapsed_seconds = flop/cpu_peak_gflops;

    std::cout << "Total floating point operations: " << flop << std::endl;
    std::cout << "Time taken: " << seconds  <<  " seconds" << std::endl;
    std::cout << "Theoretical Time taken: " << theoretical_elapsed_seconds  <<  " seconds" << std::endl;
    std::cout << "GFlops: " << flops << std::endl;
}