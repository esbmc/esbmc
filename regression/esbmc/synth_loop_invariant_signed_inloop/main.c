/* Signed counter with an assertion inside the body. A signed counter cannot
 * carry the i >= i0 conjunct, so the havoc could pick i < i0 where i - i0 is
 * negative and the closed form describes an unreachable state -- which only
 * shows up when something reads the accumulator mid-loop. Must decline, and
 * must leave the verdict alone. */
#include <assert.h>
int nondet_int();
int main(void)
{
  int n = nondet_int();
  __ESBMC_assume(n >= 1 && n <= 10);
  int i = 1, sn = 0;
  while (i <= n)
  {
    assert(sn <= 2 * n);
    sn = sn + 2;
    i++;
  }
  return 0;
}
