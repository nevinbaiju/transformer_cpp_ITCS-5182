#include <random>
#include <cmath>
#include "initializations.h"

void kaimingInit(float* array, int size, int fan_in) {
    std::random_device rd;
    std::mt19937 gen(rd());

    const float bound = sqrt(2.0 / fan_in);
    std::normal_distribution<float> distribution(0.0f, bound);

    for (int i = 0; i < size; ++i) {
        array[i] = distribution(gen);
    }
}