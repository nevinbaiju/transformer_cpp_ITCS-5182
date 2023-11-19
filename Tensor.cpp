#include <iostream>
#include <cmath>
#include <random>
#include "Tensor.h"
#include <iomanip>
#include <cstring>

Tensor::Tensor(int rows, int cols) : rows(rows), cols(cols), size(rows * cols) {
    data = new float[size];
}

Tensor::~Tensor() {
    delete[] data;
}

void Tensor::kaimingInit(int fan_in) {
    std::random_device rd;
    std::mt19937 gen(rd());

    const float bound = std::sqrt(2.0 / fan_in);
    std::normal_distribution<float> distribution(0.0f, bound);

    for (int i = 0; i < size; ++i) {
        data[i] = distribution(gen);
    }
}

void Tensor::identityInit() {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            data[i * cols + j] = (i == j) ? 1.0f : 0.0f;
        }
    }
}

void Tensor::sequentialInit() {
    for (int i = 0; i < size; ++i) {
        data[i] = i + 1;
    }
}

void Tensor::setOneInit(float val) {
    for (int i = 0; i < size; ++i) {
        data[i] = val;
    }
}

std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
    for (int i = 0; i < tensor.rows; ++i) {
        for (int j = 0; j < tensor.cols; ++j) {
            os << std::setw(8) << std::setfill(' ') << tensor.data[i * tensor.cols + j] << ", ";
        }
        os << std::endl;
    }
    return os;
}
