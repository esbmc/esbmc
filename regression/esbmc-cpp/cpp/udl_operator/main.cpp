#include <cassert>

constexpr long double operator""_deg(long double d)
{
  return d * 3.14159265358979L / 180.0L;
}

int main()
{
  constexpr long double pi = 3.14159265358979L;
  long double r = 180.0_deg;
  assert(r == pi);
  return 0;
}
