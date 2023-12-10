float* dot_product_attention(float *query, int query_rows, int query_cols,
                             float *key, int key_rows, int key_cols,
                             float *value, int value_rows, int value_cols,
                             bool scaled=true);
Tensor* dot_product_attention(Tensor *query, Tensor *key, Tensor *value, bool scaled);
                             
float* multi_head_attention(float *query, int query_rows, int query_cols,
                            float *key, int key_rows, int key_cols,
                            float *value, int value_rows, int value_cols, 
                            int num_heads, int embedding_size, bool use_embedding);
Tensor* multi_head_attention(Tensor &query, Tensor &key, Tensor &value,
                             Tensor **query_weights, Tensor **key_weights, Tensor **value_weights,
                             int num_heads, int embedding_size, bool use_embedding);                   

#ifdef CUDA
#include "D_Tensor.cuh"
D_Tensor* dot_product_attention(D_Tensor *query, D_Tensor *key, D_Tensor *value, bool scaled);
D_Tensor* multi_head_attention(D_Tensor &query, D_Tensor &key, D_Tensor &value,
                             D_Tensor **query_weights, D_Tensor **key_weights, D_Tensor **value_weights,
                             int num_heads, int embedding_size, bool use_embedding);
#endif                                                                   