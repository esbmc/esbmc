// [cmath.syn] requires the C99 additions to be visible in namespace std; the
// model only pulled in the C89 set, so std::fmin and friends did not resolve
// even though ::fmin did (github #5868).
#include <cmath>
#include <cassert>

int main()
{
  assert(std::fmin(1.0, 2.0) == 1.0);
  assert(std::fmax(1.0, 2.0) == 2.0);
  assert(std::fmin(1.0f, 2.0f) == 1.0f);
  assert(std::fmax(1.0L, 2.0L) == 2.0L);
  assert(std::fdim(5.0, 3.0) == 2.0);
  assert(std::fdim(3.0, 5.0) == 0.0);
  assert(std::hypot(3.0, 4.0) == 5.0);
  assert(std::copysign(2.0, -1.0) == -2.0);
  assert(std::remainder(5.0, 2.0) == 1.0);
  assert(std::nextafter(1.0, 2.0) > 1.0);
  assert(std::log2(8.0) == 3.0);
  assert(std::expm1(0.0) == 0.0);
  assert(std::log1p(0.0) == 0.0);
  assert(std::asinh(0.0) == 0.0);
  assert(std::acosh(1.0) == 0.0);
  assert(std::atanh(0.0) == 0.0);
  assert(std::nearbyint(2.5) == 2.0);
  assert(std::rint(3.5) == 4.0);
  return 0;
}
