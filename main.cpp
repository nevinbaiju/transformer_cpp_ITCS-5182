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
    #include "cuda_kernels.cuh"
#endif

#ifdef AVX
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

    std::uint64_t flops = (((2*n*k)/(1e6))*m)/1e3;
    std::uint64_t memory = ((m*n + m*k)*4);

    const float peak_memory_bw = 76.8;
    const float peak_flops = 1881;

    std::cout << "Time Taken: " << seconds.count() << " Flops bound: " << flops/(peak_flops*1024) << ", Memory bound: " << memory/(peak_memory_bw*1e9) << std::endl; 
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

    std::uint64_t flops = (3*2*rows*rows*cols + 4*rows*rows*cols + 3*rows*rows)/(1e9);
    std::uint64_t memory = (15*rows*cols*4);

    std::cout << output->rows << "  " << output->cols << " \n";

    const float peak_memory_bw = 76.8;
    const float peak_flops = 1881;

    std::cout << "Time Taken: " << seconds.count() << " Flops bound: " << flops/(peak_flops*1024) << ", Memory bound: " << memory/(1e9*peak_memory_bw) << std::endl; 
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
#endif

#ifdef CUDA
void test_cuda_matmul(int argc, char *argv[]){

    int n = atoi(argv[1]);
    int m = atoi(argv[2]);
    int k = atoi(argv[3]);

    D_Tensor a(n, m);
    a.setOneInit(1);
    // std::cout << a << std::endl;
    cudaDeviceSynchronize();
    // Allocate memory for b
    D_Tensor b(m, k);

    // // Initialize b before using setOneInit
    b.setOneInit(1);
    // std::cout << b;

    // // Now you can use setOneInit on b
    // b.setOneInit(1);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    D_Tensor *r = a*b;
    auto end_time = std::chrono::high_resolution_clock::now();

    Tensor *r_cpu = r->to_cpu();

    std::uint64_t flops = (((2*n*k)/(1e6))*m)/1e3;
    std::uint64_t memory = ((m*n + m*k)*4);

    const float peak_memory_bw = 76.8;
    const float peak_flops = 1881;

    std::chrono::duration<double> seconds = end_time - start_time;
    for(int i=0; i<r_cpu->size; i++){
        if (r_cpu->data[i] != m){
            std::cerr << "Ans at " << i << ": " << r_cpu->data[i] << " is wrong\n";
            // std::cout << *r;
            break;
        }
    }
    if (r->size < 50){
        std::cout << *r_cpu << std::endl;
    }

    std::cout << "Time Taken: " << seconds.count() << " Flops bound: " << flops/(peak_flops*1024) << ", Memory bound: " << memory/(peak_memory_bw*1e9) << std::endl; 
    std::cout << "Flops: " << flops/seconds.count() << std::endl;
}

void test_cuda_dotproduct_attention(int argc, char *argv[]){
    int rows = atoi(argv[1]);
    int cols = atoi(argv[2]);
    D_Tensor *a = new D_Tensor(rows, cols);
    D_Tensor *b = new D_Tensor(rows, cols);
    D_Tensor *c = new D_Tensor(rows, cols);
    a->setOneInit(1);
    b->setOneInit(1);
    c->setOneInit(1);

    D_Tensor *result = dot_product_attention(a, b, c, true);

    std::cout << *result << std::endl;;
}

void test_transpose(int argc, char * argv[]){
    D_Tensor *a = new D_Tensor(5, 9);
    D_Tensor *b = new D_Tensor(5, 9);
    a->sequentialInit();
    b->sequentialInit();

    std::vector<D_Tensor*> tenss;
    tenss.push_back(a);
    tenss.push_back(b);

    D_Tensor *res = vertical_concat(tenss);

    std::cout << *res;
}

void test_multi_head_attention(int argc, char *argv[]) {
    
    int rows = 1024;
    int64_t cols = atoi(argv[1]);
    int num_heads =  atoi(argv[2]);
    const int nb_iter = 1;
    
    D_Tensor query(rows, cols);
    D_Tensor key(rows, cols);
    D_Tensor value(rows, cols);
    
    query.setOneInit(1);
    key.setOneInit(1);
    value.setOneInit(1);
    
    int embedding_dims = cols;
    int col_split = cols/num_heads;
    
    D_Tensor **query_weights = new D_Tensor*[num_heads];
    D_Tensor **key_weights = new D_Tensor*[num_heads];
    D_Tensor **value_weights = new D_Tensor*[num_heads];

    for(int i=0; i<num_heads; i++){
        query_weights[i] = new D_Tensor(col_split, col_split);
        query_weights[i]->setOneInit(1);
        key_weights[i] = new D_Tensor(col_split, col_split);
        key_weights[i]->setOneInit(1);
        value_weights[i] = new D_Tensor(col_split, col_split);
        value_weights[i]->setOneInit(1);
    }

    D_Tensor *output;
    auto start_time = std::chrono::high_resolution_clock::now();
    #pragma unroll
    for(int i=0; i<nb_iter; i++){
        output = multi_head_attention(query, key, value, query_weights, key_weights, value_weights, num_heads, embedding_dims, true);
    }
    auto end_time = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> seconds = end_time - start_time;

    std::uint64_t flops = (3*2*rows*rows*cols + 4*rows*rows*cols + 3*rows*rows)/(1e9);
    std::uint64_t memory = (15*rows*cols*4);

    std::cout << output->rows << "  " << output->cols << " \n";

    const float peak_memory_bw = 76.8;
    const float peak_flops = 1881;

    std::cout << "Time Taken: " << seconds.count() << " Flops bound: " << flops/(peak_flops*1024) << ", Memory bound: " << memory/(1e9*peak_memory_bw) << std::endl; 
    std::cout << "Flops: " << flops/(seconds.count()*1024)<< std::endl;
    std::cout << "Time Taken: " << seconds.count()/nb_iter << " \n";
    std::cerr << cols << "," << num_heads << "," << seconds.count()/nb_iter << " \n";
    // std::cout << *output;

}

#endif

int main(int argc, char *argv[]) {
    // #ifdef AVX
    // test_matmul(argc, argv);
    // #endif 

    // #ifdef CUDA
    // test_cuda_dotproduct_attention(argc, argv);
    // #endif
    test_multi_head_attention(argc, argv);
}
