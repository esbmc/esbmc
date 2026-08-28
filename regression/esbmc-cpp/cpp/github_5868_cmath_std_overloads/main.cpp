#include <cmath>
#include <cassert>

int main()
{
  // [cmath.syn]: the C99 additions whose return type or extra parameter does
  // not fit the plain float/double/long double shape.
  assert(std::scalbn(1.5, 3) == 12.0);
  assert(std::scalbn(1.5f, 3) == 12.0f);
  assert(std::scalbln(1.5, 3L) == 12.0);

  assert(std::fma(2.0, 3.0, 4.0) == 10.0);
  assert(std::fma(2.0f, 3.0f, 4.0f) == 10.0f);

  int quo = 0;
  assert(std::remquo(10.0, 3.0, &quo) == 1.0);
  assert(quo == 3);

  assert(std::lround(2.5) == 3L);
  assert(std::llround(-2.5) == -3LL);
  assert(std::lrint(2.0) == 2L);
  assert(std::llrint(-2.0) == -2LL);
  assert(std::isnan(std::nan("")));

  // ilogb, logb and nexttoward have no model in ESBMC's libc, so only their
  // overload sets are exercised; their values are nondet.
  int e = std::ilogb(8.0) + std::ilogb(8.0f) + std::ilogb(8.0L);
  long double b = std::logb(8.0) + std::logb(8.0f) + std::logb(8.0L);
  long double t = std::nexttoward(1.0, 2.0L) + std::nexttoward(1.0f, 2.0L);
  (void)e;
  (void)b;
  (void)t;
  return 0;
}
