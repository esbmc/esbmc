/* github_6551_cond_is_fresh_pass:
 * A conditional __ESBMC_is_fresh claims nothing about the pointers on the
 * other branch, so the separation obligation must not be imposed on every
 * caller. Here n == 0 takes that branch and aliasing is legitimate. See
 * github_6551_cond_is_fresh_enforce_fail for the enforce side of the same
 * contract, which must reject it rather than consume the unstated separation. */
#include <stdlib.h>
typedef struct { int coeffs[4]; } P;
void g(P *a, P *b, int n) {
  __ESBMC_requires(n <= 0 || (__ESBMC_is_fresh(a, sizeof(P)) && __ESBMC_is_fresh(b, sizeof(P))));
  __ESBMC_ensures(1);
}
int main(void){ P *o=(P*)malloc(sizeof(P)); __ESBMC_assume(o!=0); g(o,o,0); return 0; }
