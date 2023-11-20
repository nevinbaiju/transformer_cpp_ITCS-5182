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
    std::cout << tensor1 << std::endl;
    tensor1.transpose();
    std::cout << tensor1 << std::endl;
}
