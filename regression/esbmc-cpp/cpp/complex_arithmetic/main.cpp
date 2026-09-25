// <complex> was a skeleton that could not be used at all: every member sat in
// the default-private section of `class complex`, no member or operator had a
// definition, and the comparison operators were declared to return complex<T>
// instead of bool. `std::complex<double> a(3.0, 4.0);` did not compile.
//
// All values here are exactly representable, so the floating-point equalities
// are safe to assert.
#include <complex>
#include <cmath>
#include <cassert>

int main()
{
  std::complex<double> a(3.0, 4.0);
  assert(a.real() == 3.0);
  assert(a.imag() == 4.0);

  std::complex<double> b(1.0, 2.0);

  std::complex<double> s = a + b;
  assert(s.real() == 4.0 && s.imag() == 6.0);

  std::complex<double> d = a - b;
  assert(d.real() == 2.0 && d.imag() == 2.0);

  // (3+4i)(1+2i) = -5+10i
  std::complex<double> m = a * b;
  assert(m.real() == -5.0 && m.imag() == 10.0);

  // (3+4i)/(1+2i) = (11-2i)/5
  std::complex<double> q = a / b;
  assert(q.real() == 2.2 && q.imag() == -0.4);

  std::complex<double> n = -a;
  assert(n.real() == -3.0 && n.imag() == -4.0);

  assert(std::norm(a) == 25.0);
  assert(sqrt(std::norm(a)) == 5.0); // std::abs(complex) aborts ESBMC, see
                                     // std_abs_class_overload

  std::complex<double> c = std::conj(a);
  assert(c.real() == 3.0 && c.imag() == -4.0);

  assert(std::real(a) == 3.0);
  assert(std::imag(a) == 4.0);

  // [complex.ops]: comparisons yield bool
  assert(a == std::complex<double>(3.0, 4.0));
  assert(a != b);

  std::complex<double> e(3.0, 4.0);
  e += b;
  assert(e.real() == 4.0 && e.imag() == 6.0);
  e -= b;
  assert(e == a);
  e *= 2.0;
  assert(e.real() == 6.0 && e.imag() == 8.0);
  e /= 2.0;
  assert(e == a);

  std::complex<double> f = a; // copy
  assert(f == a);
  f.real(9.0);
  assert(a.real() == 3.0); // the copy is independent

  return 0;
}
