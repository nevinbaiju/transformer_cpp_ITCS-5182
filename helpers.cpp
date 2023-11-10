#include <iostream>
#include <iomanip>
#include <cstring>

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