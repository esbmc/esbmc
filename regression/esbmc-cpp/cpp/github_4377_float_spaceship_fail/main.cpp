// Negative twin: NaN <=> x is unordered, so claiming it is greater must be
// reported. Guards against the lowering falling through to `greater`.
#include <compare>
#include <cassert>

int main()
{
  double a = 1.0;
  double nan = 0.0 / 0.0;
  auto un = nan <=> a;
  assert(un > 0);
  return 0;
}
