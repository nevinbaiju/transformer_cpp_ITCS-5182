#include <iostream>
#include <cmath>
#include <random>
#include <iomanip>
#include <cstring>

#include "Tensor.h"
#include "exceptions.h"

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
            os << tensor.data[i * tensor.cols + j] << ", ";
        }
        os << std::endl;
    }
    return os;
}

Tensor Tensor::operator*(const Tensor& other) const {
    if (cols != other.rows) {
        throw SizeMismatchException(rows, cols, other.rows, other.cols);
    }

    Tensor result(rows, other.cols);
    // std::cout << "ivide" << std::endl;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < other.rows; j++) {
            result.data[i * other.rows + j] = 0.0;
            for (int k = 0; k < cols; k++) {
                result.data[i * other.cols + j] += data[i * cols + k] * other.data[k * other.cols + j];
            }
        }
    }

    return result;
}
