/* github_6551_requires_ptr_neq_replace_fail:
 * __ESBMC_requires(p != q) is the other way to state separation, and the one
 * the docs offer when a contract needs the parameters distinct but not fresh.
 * The other direction: what enforcement rests on, a replace site has to
 * discharge, so a caller passing one object to both must be rejected. */
typedef struct { int coeffs[4]; } P;

void callee(P *a, P *b)
{
  __ESBMC_requires(a != 0);
  __ESBMC_requires(b != 0);
  __ESBMC_requires(a != b);
  __ESBMC_assigns(a->coeffs);
  __ESBMC_ensures(a->coeffs[0] == 1);
  __ESBMC_ensures(b->coeffs[0] == __ESBMC_old(b->coeffs[0]));
  a->coeffs[0] = 1;
}
int main(void) {
  P *o = (P *)__ESBMC_alloca(sizeof(P));
  o->coeffs[0] = 5;
  callee(o, o);
  __ESBMC_assert(o->coeffs[0] == 5, "implied by the second ensures");
  return 0;
}
