float* relu(float input[], int size, bool inplace = false);
float* relu(float *input, int rows, int cols, bool inplace = false);
float* softmax(float *input, int rows, int cols, bool inplace);
float* scale(float *input, int rows, int cols, bool inplace);