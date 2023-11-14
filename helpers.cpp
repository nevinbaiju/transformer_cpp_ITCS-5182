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
    double memory = ((rows*cols*cols*embedding + rows*embedding*2 + num_heads*(cols/num_heads)*(cols/num_heads)*embedding + rows*rows + rows*rows + rows*rows*rows*embedding + rows*embedding)*4)/(1024e9);
    std::chrono::duration<double> elapsed_seconds = end-start;
    double seconds = elapsed_seconds.count();
    double flops = flop/(seconds*1e9);

    double theoretical_elapsed_seconds_flops = flop/cpu_peak_gflops;
    double theoretical_memory_seconds = memory/76.8;

    std::cout << "Total floating point operations: " << flop << std::endl;
    std::cout << "Total memory operations: " << memory << " GB" << std::endl;
    
    std::cout << "Flops Time taken: " << seconds  <<  " seconds" << std::endl;
    std::cout << "Theoretical Flops Time taken: " << theoretical_elapsed_seconds_flops  <<  " seconds\n" << std::endl;


    std::cout << "Memory Time taken: " << seconds  <<  " seconds" << std::endl;
    std::cout << "Theoretical memory Time taken: " << theoretical_memory_seconds  <<  " seconds" << std::endl;
    std::cout << "GFlops: " << flops << std::endl;
}