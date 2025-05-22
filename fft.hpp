#include <complex>
#include <cstdint>
#include <span>

using std::array;
using std::span;

// Здесь реализован обычный алгоритм Кули-Тьюки, но не по основанию 2, а по
// основанию D (D = 2, 3, 5). Разница заключается лишь в том, что вместо того,
// чтобы делить массив на 2 части и рекурсивно вычислять преобразование от них,
// мы делим на D частей. Фиксируя размер преобразования N, мы даём себе
// возможность просчитать все корни из 1 во время компиляции и сэкономить время
// на умножении (а также увеличить точность, так как при повторном умножении
// ошибка накапливаетя)
template <uint32_t N, typename Complex = std::complex<double>>
class FastFourierTransform {
  using Float = typename Complex::value_type;
  constexpr static Float pi = 3.14159265359;

  constexpr static auto powers = []() {
    array<Complex, N> powers;
    for (uint32_t i = 0; i < N; ++i) {
      Float phi = -(Float)(2 * i) * pi / (Float)N;
      powers[i] = {std::cos(phi), std::sin(phi)};
    }
    return powers;
  }();

  template <bool Inverse>
  Complex GetPower(uint32_t n) {
    n %= N;
    if constexpr (Inverse) {
      n = N - n;
      n = n == N ? 0 : n;
    }
    return powers[n];
  }

  // Алгоритм заключается в сведении преобразования к нескольким меньшим. Здесь
  // М -- размер меньшего преобразования. Количество D частей, на которые мы
  // разбиваем массив, определяется тем, на что делится M.
  template <uint32_t M, bool Inverse>
  void DoTransform(span<const Complex> in, span<Complex> out) {
    if constexpr (M == 1) {
      out[0] = in[0];
      return;
    } else if constexpr (M % 2 == 0) {
      DoTransform<M, 2, Inverse>(in, out);
    } else if constexpr (M % 3 == 0) {
      DoTransform<M, 3, Inverse>(in, out);
    } else {
      static_assert(M % 5 == 0, "N кратно только 2, 3 и 5");
      DoTransform<M, 5, Inverse>(in, out);
    }
  }

  // Сам алгоритм стандартный, мы вычисляем преобразования для D частей, а затем
  // комбинируем их по схеме бабочки. Ради экономии также будем выполнять все
  // операции без дополнительных аллокаций
  template <uint32_t M, uint32_t D, bool Inverse>
  void DoTransform(span<const Complex> in, span<Complex> out) {
    static constexpr uint32_t K = M / D;
    static constexpr uint32_t stride = N / M;

    for (uint32_t i = 0; i < D; ++i) {
      DoTransform<K, Inverse>(in.subspan(i * stride), out.subspan(i * K));
    }

    for (uint32_t i = 0; i < K; ++i) {
      array<Complex, D> vals;
      for (uint32_t j = 0; j < D; ++j) {
        uint32_t idx = i + j * K;
        vals[j] = out[idx];
      }
      for (uint32_t j = 0; j < D; ++j) {
        uint32_t idx = i + j * K;
        out[idx] = 0;
        for (uint32_t k = 0; k < D; ++k) {
          uint32_t pw = stride * idx * k;
          out[idx] += vals[k] * GetPower<Inverse>(pw);
        }
      }
    }
  }

 public:
  array<Complex, N> Transform(const array<Complex, N>& input) {
    array<Complex, N> result;

    span<const Complex> in(input);
    span<Complex> out(result);

    DoTransform<N, false>(in, out);

    return result;
  }

  array<Complex, N> InverseTransform(const array<Complex, N>& input) {
    array<Complex, N> result;

    span<const Complex> in(input);
    span<Complex> out(result);

    DoTransform<N, true>(in, out);
    for (uint32_t i = 0; i < N; ++i) {
      result[i] /= N;
    }

    return result;
  }
};
