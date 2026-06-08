#include <cassert>

constexpr long double operator""_deg(long double d)
{
  return d * 3.14159265358979L / 180.0L;
}

int main()
{
  long double r = 90.0_deg;
  // 90 degrees in radians is ~1.5707..., not 90.0
  assert(r == 90.0L);
  return 0;
}
