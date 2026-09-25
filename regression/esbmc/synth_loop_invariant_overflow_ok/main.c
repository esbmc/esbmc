/* Counter and accumulator both unsigned, overflow checking on. Unsigned wrap is
 * well defined, so no claim goto_check instruments on the synthesised
 * arithmetic can invent a failure -- the decline must NOT fire here. Pins the
 * admitting side of the overflow rule: a decline that ignored the accumulator
 * types would still pass, but one that declined on any accumulator under
 * --overflow-check would lose this loop's invariant. */
#include <assert.h>
unsigned int nondet_uint();
int main(void)
{
  unsigned int n = nondet_uint();
  __ESBMC_assume(n <= 8);
  unsigned int i = 0, s = 0;
  while (i < n)
  {
    s = s + 2;
    i = i + 1;
  }
  assert(s == 2 * n);
  return 0;
}
