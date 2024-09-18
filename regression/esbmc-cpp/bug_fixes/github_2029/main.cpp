#include <cassert>
#include <cmath>

int main()
{
  double x = 1.0;
  double y = 0.0;

  assert(std::isnormal(x));
  assert(!std::isnormal(y));

  return 0;
}
