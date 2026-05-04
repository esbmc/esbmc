#include <assert.h>

int nondet_int(void);

static int compute(int a, int b)
{
  return 5 - a - 8 - b - 2;
}

int main()
{
  int x = nondet_int();

  // Same shape as the passing test, but with an off-by-one expectation —
  // must fail.
  int r = compute(10, x);
  assert(r == -14 - x);

  return 0;
}
