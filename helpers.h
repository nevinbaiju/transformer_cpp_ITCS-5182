#include <chrono>

void print_arr(float arr[], int arr_size);
void print_arr(float *arr, int rows, int cols);

void get_bench_results(int rows, int cols, int embedding, int num_heads, 
                         std::chrono::time_point<std::chrono::high_resolution_clock> start, 
                         std::chrono::time_point<std::chrono::high_resolution_clock> end);