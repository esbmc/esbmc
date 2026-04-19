#include <assert.h>

int main()
{
  int x = nondet_int();
  int y = nondet_int();
  int c1 = 1, c2 = 2;

  // x - (-y) -> x + y
  assert((x - (-y)) == (x + y));
  assert((x + y) == (x - (-y)));

  // (-x) - y -> -(x + y)
  assert(((-x) - y) == -(x + y));
  assert(-(x + y) == ((-x) - y));

  // (x - c1) - c2 → x - (c1 + c2)
  assert(((x - c1) - c2) == (x - (c1 + c2)));
  assert((x - (c1 + c2)) == ((x - c1) - c2));
  
  return 0;
}
