/* or_ensures_old_conditional_fail:
 * Same contract, but body adds delta+1 instead of delta.
 * delta==0: ensures left TRUE -> vacuously pass (cannot catch this case)
 * delta>0:  ensures left FALSE -> check *p == old(*p) + delta
 *           but body gives old + (delta+1) != old + delta -> FAIL
 *
 * ESBMC must find the delta>0 counterexample. Confirms old() inside ||
 * is correctly checked when the left side is FALSE.
 */
#include <stddef.h>

void f(int *p, int delta)
{
  __ESBMC_requires(p != NULL);
  __ESBMC_requires(delta >= 0);
  __ESBMC_ensures(delta == 0 || *p == __ESBMC_old(*p) + delta);

  *p = *p + delta + 1; /* off by one */
}

int main() { return 0; }
