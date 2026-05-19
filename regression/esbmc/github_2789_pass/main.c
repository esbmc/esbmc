// https://github.com/esbmc/esbmc/issues/2789
// Companion to github_2789: when the shift distance is constrained to the
// well-defined range [0, width), no UB is reported.
#include <assert.h>

int nondet_int();

int main()
{
  int a = nondet_int();
  __ESBMC_assume(a >= 0 && a < 32);
  assert((0 >> a) == 0);
  return 0;
}
