/* github_6542_is_fresh_vs_plain_ptr_fail:
 * __ESBMC_is_fresh(a) says a is fresh, so it is separate from everything the
 * caller can reach, not merely from other is_fresh parameters. The enforce
 * harness grants exactly that, so passing the same object as a plain pointer
 * argument has to be rejected too. */
#include <stdlib.h>
typedef struct { int coeffs[4]; } P;
void callee(P *a, P *b) {
  __ESBMC_requires(__ESBMC_is_fresh(a, sizeof(P)));
  __ESBMC_requires(b != 0);
  __ESBMC_assigns(a->coeffs);
  __ESBMC_ensures(a->coeffs[0] == 1);
  __ESBMC_ensures(b->coeffs[0] == __ESBMC_old(b->coeffs[0]));
  a->coeffs[0] = 1;
}
int main(void){ P *o=(P*)malloc(sizeof(P)); __ESBMC_assume(o!=0); o->coeffs[0]=5;
  callee(o,o); __ESBMC_assert(o->coeffs[0]==5,"false"); return 0; }
