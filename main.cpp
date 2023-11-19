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
    Tensor tensor(3, 4);

    tensor.kaimingInit(3);
    std::cout << "Kaiming Initialization:" << std::endl;
    std::cout << tensor << std::endl;

    tensor.identityInit();
    std::cout << "\nIdentity Initialization:" << std::endl;
    std::cout << tensor << std::endl;

    tensor.sequentialInit();
    std::cout << "\nSequential Initialization:" << std::endl;
    std::cout << tensor << std::endl;

    tensor.setOneInit(5.0f);
    std::cout << "\nSet One Initialization:" << std::endl;
    std::cout << tensor << std::endl;
}
