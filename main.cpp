#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <iostream>
#include <numbers>
#include <random>

#include "fft.hpp"

using Complex = std::complex<double>;

std::mt19937 gen{42};
std::normal_distribution distribution{0., 1.};

Complex GetRandom() { return {distribution(gen), distribution(gen)}; }

constexpr uint32_t N = 100000;

int main() {
  std::array<Complex, N> arr;
  for (uint32_t i = 0; i < N; ++i) {
    arr[i] = GetRandom();
  }

  FastFourierTransform<N, Complex> fft;
  auto transformed = fft.Transform(arr);

  auto reversed = fft.InverseTransform(transformed);

  double error_sum = 0;
  for (uint32_t i = 0; i < N; ++i) {
    error_sum += std::abs(arr[i] - reversed[i]);
  }

  std::cout << "Mean error: " << error_sum / N << "\n";
}
