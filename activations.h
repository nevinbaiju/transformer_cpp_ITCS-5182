#include "Tensor.h"

float* relu(float input[], int size, bool inplace = false);
float* relu(float *input, int rows, int cols, bool inplace = false);
Tensor relu(Tensor &mat, bool inplace = false);
float* softmax(float *input, int rows, int cols, bool inplace);
Tensor softmax(Tensor &mat, bool inplace = false);
float* scale(float *input, int rows, int cols, bool inplace);