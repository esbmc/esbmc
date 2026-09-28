/* Unsigned counter entering at 5 with a SYMBOLIC addend. The symbolic
 * multiplier confines the pass to the two-disjunct bound, which is only
 * establishable for an entry value of 0 (or 1 under `<=`): at entry 5 with
 * n <= 5 the loop never runs, i stays 5, and neither `i < n` nor `i == n`
 * holds. Must decline. The literal-addend regime would admit this entry
 * value via its third disjunct, so this pins the boundary between the two. */
#include <assert.h>
unsigned int nondet_uint();
int main(void)
{
  unsigned int n = nondet_uint(), a = nondet_uint();
  __ESBMC_assume(n <= 8);
  __ESBMC_assume(a <= 3);
  unsigned int i = 5, s = 0;
  while (i < n)
  {
    s = s + a;
    i = i + 1;
  }
  assert(s <= 9);
  return 0;
}
