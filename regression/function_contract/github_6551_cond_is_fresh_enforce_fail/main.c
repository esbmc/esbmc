/* github_6551_cond_is_fresh_enforce_fail:
 * A guarded __ESBMC_is_fresh states no separation on the other branch, so the
 * harness must not grant any: it keeps its allocation but stays aliasable.
 * Otherwise the callee proves its ensures from a separation the contract
 * never stated, the replace side gates its obligation on the same test and so
 * asks the caller for nothing, and enforce+replace composes to a false proof.
 * The companion _pass test pins the permissive replace side; this one pins
 * that the enforce side pays for it. */
#include <stdlib.h>
typedef struct { int coeffs[4]; } P;
void g(P *a, P *b, int n) {
  __ESBMC_requires(n <= 0 || (__ESBMC_is_fresh(a,sizeof(P)) && __ESBMC_is_fresh(b,sizeof(P))));
  __ESBMC_assigns(a->coeffs);
  __ESBMC_ensures(a->coeffs[0] == 1);
  __ESBMC_ensures(b->coeffs[0] == __ESBMC_old(b->coeffs[0]));
  a->coeffs[0] = 1;
}
int main(void){ P *o=(P*)malloc(sizeof(P)); __ESBMC_assume(o!=0); o->coeffs[0]=5;
  g(o,o,0); __ESBMC_assert(o->coeffs[0]==5,"implied by second ensures"); return 0; }
