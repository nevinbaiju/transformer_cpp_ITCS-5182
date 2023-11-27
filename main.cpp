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
    
//     std::vector<Tensor*> dot_prod_res;
    
//     for(int i=0; i<10; i++)
//         dot_prod_res.push_back(new Tensor(10, 1));

//     Tensor *result = vertical_concat(dot_prod_res);

//     std::cout << *result;
// }


int main(int argc, char *argv[]) {
    
    int rows = 100;
    int64_t cols = atoi(argv[1]);
    int num_heads =  atoi(argv[2]);
    
    Tensor query(rows, cols);
    Tensor key(rows, cols);
    Tensor value(rows, cols);
    
    query.setOneInit(1);
    key.setOneInit(1);
    value.setOneInit(1);
    
    int embedding_dims = cols;
    int col_split = cols/num_heads;
    
    Tensor **query_weights = new Tensor*[num_heads];
    Tensor **key_weights = new Tensor*[num_heads];
    Tensor **value_weights = new Tensor*[num_heads];

    for(int i=0; i<num_heads; i++){
        query_weights[i] = new Tensor(col_split, col_split);
        query_weights[i]->setOneInit(1);
        key_weights[i] = new Tensor(col_split, col_split);
        key_weights[i]->setOneInit(1);
        value_weights[i] = new Tensor(col_split, col_split);
        value_weights[i]->setOneInit(1);
    }

    auto start_time = std::chrono::high_resolution_clock::now();
    Tensor *output = multi_head_attention(query, key, value, query_weights, key_weights, value_weights, num_heads, embedding_dims, true);
    auto end_time = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> seconds = end_time - start_time;

    std::cout << output->rows << "  " << output->cols << " \n";
    std::cerr << "Time Taken: " << seconds.count() << " \n";
    std::cerr << seconds.count() << " \n";
    // std::cout << *output;

}
