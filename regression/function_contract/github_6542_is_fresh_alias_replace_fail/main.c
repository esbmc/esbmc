/* github_6542_is_fresh_alias_replace_fail:
 * __ESBMC_is_fresh states separation, and the enforce harness grants it, so a
 * replace site has to discharge it. Passing one object to both parameters used
 * to make the assumed ensures self-contradictory and discharge everything
 * after the call vacuously -- a false VERIFICATION SUCCESSFUL. */
#include <stdlib.h>

typedef struct
{
  int coeffs[4];
} P;

void callee(P *a, P *b)
{
  __ESBMC_requires(__ESBMC_is_fresh(a, sizeof(P)));
  __ESBMC_requires(__ESBMC_is_fresh(b, sizeof(P)));
  __ESBMC_assigns(a->coeffs);
  __ESBMC_ensures(a->coeffs[0] == 1);
  __ESBMC_ensures(b->coeffs[0] == __ESBMC_old(b->coeffs[0]));
  a->coeffs[0] = 1;
}

int main(void)
{
  P *o = (P *)malloc(sizeof(P));
  __ESBMC_assume(o != 0);
  o->coeffs[0] = 5;
  callee(o, o);
  __ESBMC_assert(o->coeffs[0] == 5, "implied by the second ensures");
  return 0;
}
