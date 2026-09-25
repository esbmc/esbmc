/* github_6504: an out-of-bounds read on a nondeterministically sized heap
 * object. `n` is unconstrained, so malloc(n) may return an object of one, two
 * or three bytes and the four-byte read below runs past it.
 *
 * The bounds VCC was generated and was falsifiable, but Z3 4.16.0 returned
 * unsat for it and ESBMC reported VERIFICATION SUCCESSFUL -- a false negative
 * that a default build (which ships 4.13.3) did not show. Pinned under --z3
 * because the defect lived in that backend's answer, not in the encoding.
 */
#include <stdlib.h>

int main()
{
  unsigned long n;
  int *p = (int *)malloc(n);
  __ESBMC_assume(p != 0);
  int snap = *p;
  *p = *p + 5;
  __ESBMC_assert(*p == snap + 5, "post");
  return 0;
}
