#include <iostream>
#include <cmath>
#include <random>
#include <iomanip>
#include <cstring>
#include <omp.h>

#include "Tensor.h"
#include "exceptions.h"

Tensor::Tensor(int rows, int cols) : rows(rows), cols(cols), size(rows * cols) {
    data = new float[size];
}

Tensor::~Tensor() {
    if (data != nullptr){
        delete[] data;
        data = nullptr;
    }
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

    // std::cout << "Mult" << std::endl;
    // std::cout << *this << " \n" << other << std::endl;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < other.cols; j++) {
            result.data[i * other.cols + j] = 0;
            for (int k = 0; k < cols; k++) {
                result.data[i * other.cols + j] += data[i * cols + k] * other.data[k * other.cols + j];
            }
        }
    }
    
    // std::cout << "Result: " << result << "\n end mult" << std::endl;

    return result;
}
Tensor::Tensor(const Tensor& other) : rows(other.rows), cols(other.cols), size(other.size) {
        std::memcpy(data, other.data, sizeof(float) * size);
}

void Tensor::transpose() {
    float temp;
    float *result = new float[rows*cols];
    // std::cout << "Transposing \n" << *this << std::endl;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[j*rows + i] = data[i*cols + j];
        }
    }
    delete[] data;
    data = result;

    temp = rows;
    rows = cols;
    cols = temp;
}

// Need to change this to 1D
Tensor** Tensor::vertical_split(int num_splits) {

    int splitSize = cols / num_splits;
    
    Tensor **result = new Tensor*[num_splits];
    for (int i = 0; i < num_splits; ++i) {
        result[i] = new Tensor(rows, splitSize);
    }
    
    for (int i = 0; i < num_splits; ++i) {
        int col_offset = i * splitSize;
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < splitSize; ++col) {
                result[i]->data[row * splitSize + col] = data[row * cols + col_offset + col];
            }
        }
    }

    return result;
}