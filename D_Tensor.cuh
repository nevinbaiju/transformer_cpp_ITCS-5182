#ifndef TENSOR_D
#define TENSOR_D

#include <cuda_runtime.h>
#include "Tensor.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

class D_Tensor {
public:
    D_Tensor(int rows, int cols);
    ~D_Tensor();

    int rows;
    int cols;
    int size;
    int block_size=256;
    float* data;

    // void kaimingInit(int fan_in);
    // void identityInit();
    void sequentialInit();
    void setOneInit(float val);
    Tensor* to_cpu();
    void transpose();
    // D_Tensor* operator*(const D_Tensor& other) const;
    // D_Tensor** vertical_split(int num_splits);
    // D_Tensor(const D_Tensor& other); 

    friend std::ostream& operator<<(std::ostream& os, const D_Tensor& tensor);
};

#endif