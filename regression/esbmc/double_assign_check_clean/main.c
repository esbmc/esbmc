#include <assert.h>

int nondet_int(void);

int accumulate(int n)
{
  int sum = 0;
  for (int i = 0; i < 4; i++)
    sum += n + i;
  return sum;
}

int main(void)
{
  // A branch, a loop and two calls to one function exercise phi emission and
  // L1 re-activation, the two shapes that could plausibly redefine an SSA name.
  int x = nondet_int();
  __ESBMC_assume(x > 0 && x < 100);
  int y = accumulate(x); // 4x + 6, so 10 < y < 402
  if (nondet_int() > 0)
    y = accumulate(y);
  assert(y > 0);
  return 0;
}
