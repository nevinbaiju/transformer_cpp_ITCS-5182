#include <random>
#include <cmath>

void kaimingInit(float** array, int rows, int cols, int fan_in);
void setOneInit(float* array, int rows, int cols, int val=1);
void sequentialInit(float *array, int rows, int cols);
float* transpose(float* array, int rows, int cols, bool inplace=false);