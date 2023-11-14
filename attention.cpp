#include "computations.h"
#include "activations.h"
#include "helpers.h"
#include "initializations.h"

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

    float *output;
    output = matmul(attention_weights, query_rows, 
                    key_rows, value, value_rows, value_cols);

    return output;
}

float* multi_head_attention(float *query, int query_rows, int query_cols,
                            float *key, int key_rows, int key_cols,
                            float *value, int value_rows, int value_cols, 
                            int num_heads, int embedding_size, bool use_embedding){
    int split_col_size = query_cols/num_heads;

    float **query_split = vertical_split(query, query_rows, query_cols, num_heads);
    float **key_split = vertical_split(key, key_rows, key_cols, num_heads);
    float **value_split = vertical_split(value, value_rows, value_cols, num_heads);
    float *result = new float[query_rows*value_cols];
    
    if (use_embedding){
        float **query_embedding_weights, **key_embedding_weights, **value_embedding_weights;
        for(int head=0; head<num_heads; head++){
           query_embedding_weights[head] = new float[split_col_size*split_col_size];
           identityInit(query_embedding_weights[head], split_col_size, split_col_size);
           query_split[head] = matmul(query_split[head], query_rows, split_col_size, 
                                      query_embedding_weights[head], split_col_size, split_col_size);

           key_embedding_weights[head] = new float[key_rows*split_col_size];
           identityInit(key_embedding_weights[head], split_col_size, split_col_size);
           key_split[head] = matmul(key_split[head], key_rows, split_col_size, 
                                      key_embedding_weights[head], split_col_size, split_col_size);

           value_embedding_weights[head] = new float[value_rows*split_col_size];
           identityInit(value_embedding_weights[head], split_col_size, split_col_size);
           value_split[head] = matmul(value_split[head], value_rows, split_col_size, 
                                      value_embedding_weights[head], split_col_size, split_col_size);
        }
    }
    float *temp_output;
    int col_offset;
    for(int head=0; head<num_heads; head++){
        temp_output = dot_product_attention(query_split[head], query_rows, split_col_size,
                                            key_split[head], key_rows, split_col_size,
                                            value_split[head], value_rows, split_col_size, true);
        for(int i=0; i<query_rows; i++){
            for(int j=0; j<split_col_size; j++){
                col_offset = head*split_col_size;
                result[i*value_cols + col_offset+j] = temp_output[i*split_col_size + j];
            }
        }
    }

    return result;
}