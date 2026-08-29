/* github_6551_requires_ptr_neq_pass:
 * __ESBMC_requires(p != q) is the other way to state separation, and the one
 * the docs offer when a contract needs the parameters distinct but not fresh.
 * Stating it must be enough to make the contract provable under enforcement,
 * where the parameters are otherwise free to alias. */
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
int main(void) { return 0; }
