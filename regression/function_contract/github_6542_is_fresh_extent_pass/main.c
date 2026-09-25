/* github_6542_is_fresh_extent_pass:
 * __ESBMC_is_fresh(p, n) asks for n bytes, and a replace site has to hold the
 * caller to it. The extent operand used to be dropped in the lowering, which
 * The control: an extent the caller does satisfy must still compose, so the
 * check has to be an implication about size and not a blanket rejection. */
#include <stdlib.h>
typedef struct { int coeffs[4]; } P;

void callee(P *p) {
  __ESBMC_requires(__ESBMC_is_fresh(p, sizeof(P)));
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
