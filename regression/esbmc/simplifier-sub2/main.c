#include <assert.h>

int main()
{
  int x = nondet_int();
  int y = nondet_int();

  // x - (-y) → x + y
  assert((x - (-y)) == (x + y));

  return 0;
}
