#include <iostream>
#include <vector>
#include "activations.h"
#include "initializations.h"
#include "helpers.h"
#include "computations.h"

int main(int argc, char *argv[]) {
    float *arr = new float[50];
    
    for(int i=0; i<50; i++){
        arr[i] = i;
    }
    print_arr(arr, 5, 10);
    transpose(arr, 5, 10, true);
    print_arr(arr, 10, 5);  

    return 0;
}
