/* github_6542_is_fresh_param_extent_pass:
 * The control for github_6542_is_fresh_param_extent_fail. The extent still
 * names a parameter, but the caller supplies an object that satisfies it, so
 * the check has to be an implication about size and not a blanket rejection
 * of extents that are not compile-time constants. */
#include <stdlib.h>

void callee(unsigned char *b, unsigned long n) {
  __ESBMC_requires(__ESBMC_is_fresh(b, n));
  __ESBMC_assigns(b[0]);
  __ESBMC_ensures(b[0] == 1);
  b[0] = 1;
}

int main(void) {
  unsigned char *p = (unsigned char *)malloc(16);
  __ESBMC_assume(p != 0);
  callee(p, 16);
  return 0;
}
