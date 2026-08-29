/* github_6542_is_fresh_interior_ptr_fail:
 * An interior pointer has a non-zero offset, so the room left in the object is
 * what follows it, not the whole allocation. Dropping the offset term would
 * accept &a[3] against an extent that only the base could satisfy. */
#include <stdlib.h>

void callee(unsigned char *b) {
  __ESBMC_requires(__ESBMC_is_fresh(b, 8));
  __ESBMC_assigns(b[0]);
  __ESBMC_ensures(b[0] == 1);
  b[0] = 1;
}

int main(void) {
  unsigned char *p = (unsigned char *)malloc(10);
  __ESBMC_assume(p != 0);
  callee(&p[3]);
  return 0;
}
