#include <iostream>

class Tensor {
public:
    Tensor(int rows, int cols);
    ~Tensor();

    void kaimingInit(int fan_in);
    void identityInit();
    void sequentialInit();
    void setOneInit(float val);

    friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor);

private:
    int rows;
    int cols;
    int size;
    float* data;
};