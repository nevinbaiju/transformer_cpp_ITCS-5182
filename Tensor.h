#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>

class Tensor {
public:
    Tensor(int rows, int cols);
    ~Tensor();

    int rows;
    int cols;
    int size;
    float* data;

    void kaimingInit(int fan_in);
    void identityInit();
    void sequentialInit();
    void setOneInit(float val);
    void transpose();
    Tensor* operator*(const Tensor& other) const;
    Tensor** vertical_split(int num_splits);
    Tensor(const Tensor& other); 

    friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor);
};

#endif