#include <cstring>
#include <vector>

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

Tensor* dot_product_attention(Tensor &query, Tensor &key, Tensor &value, bool scaled){
    key.transpose();
    Tensor attention_weights = query * key;
    
    if (scaled){
        scale(attention_weights, true);
    }
    softmax(attention_weights, true);
    
    Tensor _out = attention_weights * value;
    Tensor *out = new Tensor(attention_weights.rows, value.cols);
    std::memcpy(out->data, _out.data, sizeof(float) * _out.size);

    return out;
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



Tensor* multi_head_attention(Tensor &query, Tensor &key, Tensor &value,
                             Tensor **query_weights, Tensor **key_weights, Tensor **value_weights,
                             int num_heads, int embedding_size, bool use_embedding){
    int split_col_size = query.cols/num_heads;

    Tensor **query_split = query.vertical_split(num_heads);
    Tensor **key_split = key.vertical_split(num_heads);
    Tensor **value_split = value.vertical_split(num_heads);
    
    Tensor **query_transformed = new Tensor*[num_heads];
    Tensor **key_transformed = new Tensor*[num_heads];
    Tensor **value_transformed = new Tensor*[num_heads];

    if(use_embedding){
        for(int head=0; head<num_heads; head++){
            Tensor query_temp = *query_split[head] * *query_weights[head];
            query_transformed[head] = new Tensor(query_temp.rows, query_temp.cols);
            std::memcpy(query_transformed[head]->data, query_temp.data, sizeof(float) * query_temp.size);

            Tensor key_temp = *key_split[head] * *key_weights[head];
            key_transformed[head] = new Tensor(key_temp.rows, key_temp.cols);
            std::memcpy(key_transformed[head]->data, key_temp.data, sizeof(float) * key_temp.size);

            Tensor value_temp = *value_split[head] * *value_weights[head];
            value_transformed[head] = new Tensor(value_temp.rows, value_temp.cols);
            std::memcpy(value_transformed[head]->data, value_temp.data, sizeof(float) * value_temp.size);
        }
    }

    std::vector<Tensor*> dot_prod_res;

    for(int head=0; head<num_heads; head++){
        dot_prod_res.push_back(dot_product_attention(*query_transformed[head], *key_transformed[head], *value_transformed[head], true));
    }

    return vertical_concat(dot_prod_res);
}