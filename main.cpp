#include <iostream>
#include <vector>
#include <chrono>

#include "activations.h"
#include "initializations.h"
#include "helpers.h"
#include "computations.h"
#include "attention.h"
#include "Tensor.h"

// int main(int argc, char *argv[]) {
//     Tensor query(1, 1);
//     Tensor key(1, 10);
//     Tensor value(1, 10);
    
//     query.setOneInit(1);
//     key.setOneInit(1);
    
//     std::cout << query * key;
// }


int main(int argc, char *argv[]) {
    Tensor query(10, 1000);
    Tensor key(10, 1000);
    Tensor value(10, 1000);
    
    query.setOneInit(1);
    key.setOneInit(1);
    value.setOneInit(1);

    Tensor *output = dot_product_attention(query, key, value, true);
    std::cout << *output;
}
