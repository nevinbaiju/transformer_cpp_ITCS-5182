#include <random>
#include <cmath>
#include <vector>

#include "Tensor.h"
#ifdef CUDA
#include "D_Tensor.cuh"
#endif

void kaimingInit(float** array, int rows, int cols, int fan_in);
void setOneInit(float* array, int rows, int cols, int val=1);
void sequentialInit(float *array, int rows, int cols);
void identityInit(float *array, int rows, int cols);

float* transpose(float* array, int rows, int cols, bool inplace=false);
float** vertical_split(float *matrix, int rows, int cols, int num_splits);

Tensor *vertical_concat(std::vector<Tensor*> tensors);
#ifdef CUDA
D_Tensor *vertical_concat(std::vector<D_Tensor*> tensors);
#endif