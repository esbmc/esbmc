/* Soundness: Path 2 (step recognition) must bail when a CFG join
 * feeds the loop head. Two branches assign different inits (i=0 vs
 * i=9) and the loop's post-value differs between paths — 12 from
 * init=0, 13 from init=9. A naive parse_init would pick the textually
 * nearest ASSIGN (i=9), commit to post-value 13, and silently drop
 * the init=0 path, masking the bug. */
#include <assert.h>
int nondet_int();
int main()
{
  int i;
  int cond = nondet_int();
  if (cond)
    i = 0;
  else
    i = 9;
  for (; i < 10; i += 4)
    ;
  /* When cond is true (init=0), the real loop exits at i=12, not 13.
   * The assertion must fail on that branch. */
  __ESBMC_assert(i == 13, "post-value differs by entry path");
  return 0;
}
