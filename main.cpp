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
    tensor1.setOneInit(1);
    Tensor tensor2(4, 4);
    tensor2.setOneInit(1);

    Tensor result = tensor1 * tensor2;

    std::cout << result << std::endl;
}
