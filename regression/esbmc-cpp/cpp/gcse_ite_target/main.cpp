#include <cassert>

int nondet_int();

// `(c ? x : y) = 5` writes x or y, so `x + 1` must be recomputed.
int main()
{
  int x = 1, y = 1;
  int c = nondet_int();
  int a = x + 1;
  (c ? x : y) = 5;
  int b = x + 1;
  assert(b == (c ? 6 : 2));
  return a;
}
