#include <iostream>
#include <vector>
#include "activations.h"
#include "initializations.h"
#include "helpers.h"

int main() {
    float *input = new float[10];
    float *result;
    int size=10;

    for(int i=0; i<10; i++){
        input[i] = -5+i;
    }

    print_arr(input, size);
    result = relu(input, size, false);
    print_arr(input, size);
    print_arr(result, size);

    return 0;
}
