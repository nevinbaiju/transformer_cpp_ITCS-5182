#include <iostream>
#include <vector>
#include <chrono>

#include "activations.h"
#include "initializations.h"
#include "helpers.h"
#include "computations.h"
#include "attention.h"
#include "Tensor.h"

int main(int argc, char *argv[]) {
    Tensor tensor1(3, 4);
    tensor1.sequentialInit();

    // Using copy constructor to create tensor2 as a copy of tensor1
    Tensor tensor2 = tensor1;
    tensor2.transpose();
    // Displaying the content of tensor1 and tensor2 to show they are separate entities
    std::cout << tensor1 << std::endl;

    std::cout << tensor2 << std::endl;
}
