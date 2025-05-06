#include <cmath>
#include <complex>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <span>

using namespace std::complex_literals;

template <uint32_t N, typename Complex = std::complex<double>>
class FastFourierTransform {
  using Float = typename std::complex<double>::value_type;
  constexpr static Float pi = 3.14159265359;

  constexpr static auto powers = []() {
    std::array<Complex, N> powers;
    for (uint32_t i = 0; i < N; ++i) {
      Float phi = (Float)(2 * i) * pi / (Float)N;
      powers[i] = {std::cos(phi), std::sin(phi)};
    }
    return powers;
  }();

  Complex GetPower(uint32_t n, bool inverse) {
    n %= N;
    if (inverse) {
      n = N - n;
      n = n ? N - n : 0;
    }
    return powers[n];
  }

  template <uint32_t M, uint32_t D, bool Inverse>
  void DoTransform(std::span<const Complex> in, std::span<Complex> out, uint32_t base_power) {
    static constexpr uint32_t K = M / D;
    static constexpr uint32_t stride = N / M;

    for (uint32_t i = 0; i < D; ++i) {
      DoTransform<K, Inverse>(in.subspan(i * stride), out.subspan(i * stride), base_power + i * stride);
    }

    for (uint32_t i = 0; i < K; ++i) {
      std::array<Complex, D> vals;
      for (uint32_t j = 0; j < D; ++j) {
        uint32_t idx = (i * D + j) * stride;
        vals[j] = out[idx];
      }
      for (uint32_t j = 0; j < D; ++j) {
        uint32_t idx = (i * D + j) * stride;
        out[idx] = 0;
        for (uint32_t k = 0; k < D; ++k) {
          uint32_t pw = idx * k;
          out[idx] += vals[k] * GetPower(pw, Inverse);
        }
      }
    }
  }

  template <uint32_t M, bool Inverse>
  void DoTransform(std::span<const Complex> in, std::span<Complex> out, uint32_t base_power) {
    if constexpr (M == 1) {
      out[0] = in[0];
      return;
    } else if constexpr (M % 2 == 0) {
      DoTransform<M, 2, Inverse>(in, out, base_power);
    } else if constexpr (M % 3 == 0) {
      DoTransform<M, 3, Inverse>(in, out, base_power);
    } else {
      static_assert(M % 5 == 0, "N кратно только 2, 3 и 5");
      DoTransform<M, 5, Inverse>(in, out, base_power);
    }
  }

 public:
  std::array<Complex, N> Transform(const std::array<Complex, N>& input) {
    std::array<Complex, N> result;

    std::span<const Complex> in(input);
    std::span<Complex> out(result);

    DoTransform<N, false>(in, out, 0);

    return result;
  }

  std::array<Complex, N> InverseTransform(const std::array<Complex, N>& input) {
    std::array<Complex, N> result;

    std::span<const Complex> in(input);
    std::span<Complex> out(result);

    DoTransform<N, true>(in, out, 0);
    for (uint32_t i = 0; i < N; ++i) {
      result[i] /= N;
    }

    return result;
  }
};

using Complex = std::complex<double>;

std::mt19937 gen{42};
std::normal_distribution distribution{0., 1.};

Complex GetRandom() { return {distribution(gen), distribution(gen)}; }

constexpr uint32_t N = 10000;

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
    error_sum += std::abs(arr[i] - transformed[i]);
  }

  std::cout << "Mean error: " << error_sum / N << "\n";
}
