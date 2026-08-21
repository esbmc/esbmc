#include <cmath>
#include <cassert>

int main()
{
  int mant = 3;
  assert(std::ldexp(mant, 2) == 13.0);
  return 0;
}
