#include "computations.h"
#include "activations.h"
#include "helpers.h"

float* dot_product_attention(float *query, int query_rows, int query_cols,
                             float *key, int key_rows, int key_cols,
                             float *value, int value_rows, int value_cols, 
                             bool scaled){
    
    float *attention_weights = new float[query_rows*key_cols];
    attention_weights = matmul(query, query_rows, query_cols,
                               key, key_rows, key_cols);

    if (scaled){
        scale(attention_weights, query_rows, key_rows, true);
    }

    softmax(attention_weights, query_rows, key_rows, true);

    float *output = new float[query_rows*value_cols];
    output = matmul(attention_weights, query_rows, 
                    key_rows, value, value_rows, value_cols);

    return output;
}