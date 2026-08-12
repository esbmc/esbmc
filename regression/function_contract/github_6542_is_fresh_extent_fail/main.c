/* github_6542_is_fresh_extent_fail:
 * __ESBMC_is_fresh(p, n) asks for n bytes, and a replace site has to hold the
 * caller to it. The extent operand used to be dropped in the lowering, which
 * left `valid_object(p)` alone standing -- so a 16-byte object discharged a
 * request for 4096. */
#include <stdlib.h>
typedef struct { int coeffs[4]; } P;

void callee(P *p) {
  __ESBMC_requires(__ESBMC_is_fresh(p, 4096));
  __ESBMC_assigns(p->coeffs);
  __ESBMC_ensures(p->coeffs[0] == 1);
  p->coeffs[0] = 1;
}

int main(void) {
  P *o = (P *)malloc(sizeof(P));
  __ESBMC_assume(o != 0);
  callee(o);
  return 0;
}
