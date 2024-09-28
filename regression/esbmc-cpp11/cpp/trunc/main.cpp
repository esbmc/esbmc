#include <cmath>
#include <cassert>

int main()
{
  float f1 = 3.99f;
  float f2 = -2.99f;
  float f3 = 0.0f;

  assert(std::trunc(f1) == 3.0f);
  assert(std::trunc(f2) == -2.0f);
  assert(std::trunc(f3) == 0.0f);

  double d1 = 5.99;
  double d2 = -5.01;
  double d3 = 123456789.123456;

  assert(std::trunc(d1) == 5.0);
  assert(std::trunc(d2) == -5.0);
  assert(std::trunc(d3) == 123456789.0);

  long double ld1 = 9.99L;
  long double ld2 = -9.999L;
  long double ld3 = 0.12345L;

  assert(std::trunc(ld1) == 9.0L);
  assert(std::trunc(ld2) == -9.0L);
  assert(std::trunc(ld3) == 0.0L);

  return 0;
}
