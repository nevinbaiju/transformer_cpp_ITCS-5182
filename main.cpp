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
    int n = atoi(argv[1]);
    int m = atoi(argv[2]);
    int k = atoi(argv[3]);
    Tensor a(n, m);
    Tensor b(m, k);
    a.setOneInit(1);
    b.setOneInit(1);

    auto start_time = std::chrono::high_resolution_clock::now();
    Tensor r = a*b;
    auto end_time = std::chrono::high_resolution_clock::now();

    std::cout << r.data[0] << std::endl;
    std::chrono::duration<double> seconds = end_time - start_time;
    std::cout << "Time Taken: " << seconds.count() << " \n";
    std::cerr << seconds.count() << " \n";

    std::int64_t flops = (2*n*k*m - n*k)/(1024e3);
    std::uint64_t memory = ((m*n +m*n*k + m*k)*4)/(1024e3);

    const float peak_memory_bw = 76.8;
    const float peak_flops = 1881;

    std::cout << "Flops bound: " << flops/peak_flops << ", Memory bound: " << memory/peak_memory_bw << std::endl; 
}


// int main(int argc, char *argv[]) {
    
//     int rows = 128;
//     int64_t cols = atoi(argv[1]);
//     int num_heads =  atoi(argv[2]);
    
//     Tensor query(rows, cols);
//     Tensor key(rows, cols);
//     Tensor value(rows, cols);
    
//     query.setOneInit(1);
//     key.setOneInit(1);
//     value.setOneInit(1);
    
//     int embedding_dims = cols;
//     int col_split = cols/num_heads;
    
//     Tensor **query_weights = new Tensor*[num_heads];
//     Tensor **key_weights = new Tensor*[num_heads];
//     Tensor **value_weights = new Tensor*[num_heads];

//     for(int i=0; i<num_heads; i++){
//         query_weights[i] = new Tensor(col_split, col_split);
//         query_weights[i]->setOneInit(1);
//         key_weights[i] = new Tensor(col_split, col_split);
//         key_weights[i]->setOneInit(1);
//         value_weights[i] = new Tensor(col_split, col_split);
//         value_weights[i]->setOneInit(1);
//     }

//     auto start_time = std::chrono::high_resolution_clock::now();
//     Tensor *output = multi_head_attention(query, key, value, query_weights, key_weights, value_weights, num_heads, embedding_dims, true);
//     auto end_time = std::chrono::high_resolution_clock::now();

//     std::chrono::duration<double> seconds = end_time - start_time;

//     std::cout << output->rows << "  " << output->cols << " \n";
//     std::cout << "Time Taken: " << seconds.count() << " \n";
//     std::cerr << seconds.count() << " \n";
//     // std::cout << *output;

// }
