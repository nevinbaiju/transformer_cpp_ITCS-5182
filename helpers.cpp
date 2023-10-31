#include <iostream>

void print_arr(float arr[], int arr_size){
    for(int i=0; i<arr_size; i++){
        std::cout << arr[i] << ", ";
    }
    std::cout << "\n";
}