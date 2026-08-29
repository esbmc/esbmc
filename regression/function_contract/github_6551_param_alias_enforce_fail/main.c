/* github_6551_param_alias_enforce_fail:
 * The harness used to back each pointer parameter separately, letting a proof
 * rest on the parameters addressing distinct objects -- a hypothesis no clause
 * states and nothing checks at a replace site. Composing enforce and replace
 * then proved properties false in the real program. Parameters may now alias,
 * so this contract must be rejected: with a == b the first ensures is false. */
#include <stdlib.h>

typedef struct
{
  int coeffs[4];
} P;

void callee(P *a, P *b)
{
  __ESBMC_requires(a != 0);
  __ESBMC_requires(b != 0);
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
