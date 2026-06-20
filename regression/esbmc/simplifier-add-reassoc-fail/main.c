#include <assert.h>

int nondet_int(void);

int main()
{
  int x = nondet_int();
  int y = nondet_int();

  // Same reassoc shape as the passing test, but with a wrong expected value.
  // (x + y) + (-x) reassociates to y; assert against y+1 must fail.
  assert(((x + y) + (-x)) == (y + 1));

  return 0;
}
