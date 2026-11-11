// #9405: partial_sort must leave the smallest mid - first elements, sorted,
// in [first, mid).
#include <algorithm>
#include <cassert>

int nondet_int();

int main()
{
  int a[3] = {2, 0, 1};
  std::partial_sort(a, a + 2, a + 3);
  assert(a[0] == 0 && a[1] == 1 && a[2] == 2);

  int x = nondet_int(), y = nondet_int(), z = nondet_int();
  __ESBMC_assume(x > -100 && x < 100 && y > -100 && y < 100);
  __ESBMC_assume(z > -100 && z < 100);
  int b[3] = {x, y, z};
  std::partial_sort(b, b + 2, b + 3);
  assert(b[0] <= b[1] && b[1] <= b[2]);
  assert(b[0] + b[1] + b[2] == x + y + z);
  return 0;
}
