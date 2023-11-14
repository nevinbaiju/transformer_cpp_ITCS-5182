#include <exception>
#include <string>
#include <sstream>

class SizeMismatchException : public std::exception {
public:
    SizeMismatchException(int r1, int c1, int r2, int c2);

    const char* what() const noexcept override;

private:
    std::string message_;
};

class InvalidDimensionException : public std::exception {
public:
    InvalidDimensionException(int r1, int c1, int r2, int c2);

    const char* what() const noexcept override;

private:
    std::string message_;
};