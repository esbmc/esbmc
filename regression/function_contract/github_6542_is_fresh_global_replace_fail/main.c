/* github_6542_is_fresh_global_replace_fail:
 * __ESBMC_is_fresh is not only for parameters. The enforce harness gives a
 * global, an s->p or an *out its own allocation exactly as it does a bare
 * parameter, so a replace site owes the same separation for them. Keying the
 * obligation on a parameter position emitted nothing for these forms, leaving
 * the composition hole open for the primitive that is meant to close it. */
#include <stdlib.h>
typedef struct { int coeffs[4]; } P;
P *g;
/* is_fresh 用在全局上: enforce 侧单独分配, replace 侧原本不发义务 */
void callee(P *a) {
  __ESBMC_requires(__ESBMC_is_fresh(a, sizeof(P)));
  __ESBMC_requires(__ESBMC_is_fresh(g, sizeof(P)));
  __ESBMC_assigns(a->coeffs);
  __ESBMC_ensures(a->coeffs[0] == 1);
  __ESBMC_ensures(g->coeffs[0] == __ESBMC_old(g->coeffs[0]));
  a->coeffs[0] = 1;
}
int main(void){ P *o=(P*)malloc(sizeof(P)); __ESBMC_assume(o!=0); g=o; o->coeffs[0]=5;
  callee(o); __ESBMC_assert(o->coeffs[0]==5,"false"); return 0; }
