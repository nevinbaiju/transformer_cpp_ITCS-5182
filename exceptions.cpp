#include <iostream>
#include <exception>
#include <sstream>
#include "exceptions.h"

SizeMismatchException::SizeMismatchException(int r1, int c1, int r2, int c2) {
    std::stringstream message;
    message << "Size Mismatch: Matrices of size (" << r1 << ", " << c1 << ") and (" << r2 << ", " << c2 << ") cannot be broadcast together";
    message_ = message.str();
}

const char* SizeMismatchException::what() const noexcept {
    return message_.c_str();
}

// InvalidDimensionException::InvalidDimensionException(int rows, int cols) {
//     std::stringstream message;
//     message << "Given size of Rows: " << rows << "Does not match with the given size of columns: " << cols << std::endl;
//     message_ = message.str();
// }

// const char* InvalidDimensionException::what() const noexcept {
//     return message_.c_str();
// }
