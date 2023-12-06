#include <iostream>
#include <vector>
#include <chrono>

#include "activations.h"
#include "initializations.h"
#include "helpers.h"
#include "computations.h"
#include "attention.h"
#include "Tensor.h"

#ifdef CUDA
    #include "D_Tensor.cuh"
#endif

void test_matmul(int argc, char *argv[]) {
    int n = atoi(argv[1]);
    int m = atoi(argv[2]);
    int k = atoi(argv[3]);
    Tensor a(n, m);
    Tensor b(m, k);
    a.setOneInit(1);
    b.setOneInit(1);

    auto start_time = std::chrono::high_resolution_clock::now();
    Tensor *r = a*b;
    auto end_time = std::chrono::high_resolution_clock::now();

    for(int i=0; i<r->size; i++){
        if (r->data[i] != m){
            std::cerr << "Ans at " << i << ": " << r[i] << " is wrong\n";
        }
    }
    std::cout << "Result verified succesfully" << std::endl;
    std::chrono::duration<double> seconds = end_time - start_time;
    std::cerr << seconds.count() << " \n";

    std::uint64_t flops = (((2*n*k)/(1024e2))*m)/1024;
    std::uint64_t memory = ((m*n + m*k)*4)/(1024e3);

    const float peak_memory_bw = 76.8;
    const float peak_flops = 1881;

    std::cout << "Time Taken: " << seconds.count() << " Flops bound: " << flops/(peak_flops*1024) << ", Memory bound: " << memory/peak_memory_bw << std::endl; 
    std::cout << "Flops: " << flops/seconds.count() << std::endl;
}


void test_multi_head_attention(int argc, char *argv[]) {
    
    int rows = 1024;
    int64_t cols = atoi(argv[1]);
    int num_heads =  atoi(argv[2]);
    const int nb_iter = 1;
    
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

    Tensor *output;
    auto start_time = std::chrono::high_resolution_clock::now();
    #pragma unroll
    for(int i=0; i<nb_iter; i++){
        output = multi_head_attention(query, key, value, query_weights, key_weights, value_weights, num_heads, embedding_dims, true);
    }
    auto end_time = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> seconds = end_time - start_time;

    std::uint64_t flops = (3*2*rows*rows*cols + 4*rows*rows*cols + 3*rows*rows)/(1024e2);
    std::uint64_t memory = (6*rows*cols*4)/(1024e3);

    std::cout << output->rows << "  " << output->cols << " \n";

    const float peak_memory_bw = 76.8;
    const float peak_flops = 1881;

    std::cout << "Time Taken: " << seconds.count() << " Flops bound: " << flops/(peak_flops*1024) << ", Memory bound: " << memory/peak_memory_bw << std::endl; 
    std::cout << "Flops: " << flops/(seconds.count()*1024)<< std::endl;
    std::cout << "Time Taken: " << seconds.count()/nb_iter << " \n";
    std::cerr << cols << "," << num_heads << "," << seconds.count()/nb_iter << " \n";
    // std::cout << *output;

}


void test_scale(int argc, char *argv[]){
    Tensor *a = new Tensor(3, 4);
    a->setOneInit(1);

    softmax(a, true);

    std::cout << *a << std::endl;;
}

void test_dot_attention(int argc, char *argv[]){
    Tensor *a = new Tensor(8, 8);
    Tensor *b = new Tensor(8, 8);
    Tensor *c = new Tensor(8, 8);
    a->setOneInit(1);
    b->setOneInit(1);
    c->setOneInit(1);

    Tensor *result = dot_product_attention(a, b, c, true);

    std::cout << *result << std::endl;;
}


#ifdef CUDA
void test_cuda(int argc, char *argv[]){
    D_Tensor b(2, 3);
    b.setOneInit(1);

    D_Tensor a(2, 3);
    // // b.setOneInit(1);
    
    std::cout << b ;
}
#endif

int main(int argc, char *argv[]) {
    test_multi_head_attention(argc, argv);
}
