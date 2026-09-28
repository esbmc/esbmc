// std::min over two doubles must return the smaller double, not a truncated int.
#include <algorithm>
#include <cassert>

double nondet_double();

int main()
{
  assert(std::min(0.5, 2.0) == 0.5);
  assert(std::min(-0.25, 0.75) == -0.25);

  double a = nondet_double(), b = nondet_double();
  __ESBMC_assume(a > 0.25 && a < 0.75 && b > 1.0 && b < 2.0);
  assert(std::min(a, b) == a);
  return 0;
}
