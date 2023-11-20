#include <iostream>

class Tensor {
public:
    Tensor(int rows, int cols);
    ~Tensor();

    int rows;
    int cols;
    int size;

    void kaimingInit(int fan_in);
    void identityInit();
    void sequentialInit();
    void setOneInit(float val);
    void transpose();
    Tensor operator*(const Tensor& other) const;
    Tensor(const Tensor& other); 

    friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor);

private:
    float* data;
};