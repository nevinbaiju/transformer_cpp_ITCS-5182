float* dot_product_attention(float *query, int query_rows, int query_cols,
                             float *key, int key_rows, int key_cols,
                             float *value, int value_rows, int value_cols,
                             bool scaled=true);
Tensor dot_product_attention(Tensor &query, Tensor &key, Tensor &value, bool scaled);
                             
float* multi_head_attention(float *query, int query_rows, int query_cols,
                            float *key, int key_rows, int key_cols,
                            float *value, int value_rows, int value_cols, 
                            int num_heads, int embedding_size, bool use_embedding);                             