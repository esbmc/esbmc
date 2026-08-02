// Non-vacuity guard for complex_arithmetic: the multiplication really computes,
// so a wrong expectation must FAIL.
#include <complex>
#include <cassert>

int main()
{
  std::complex<double> a(3.0, 4.0), b(1.0, 2.0);
  std::complex<double> m = a * b;
  assert(m.real() == 5.0); // the real part is -5.0
  return 0;
}
