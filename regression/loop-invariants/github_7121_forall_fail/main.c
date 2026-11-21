/* #7121: the body overwrites coeffs[N - 1] on the first iteration, so the
 * elements not yet processed no longer equal their values on entry. */
#define N 4
typedef struct
{
  short coeffs[N];
} poly;
unsigned j;

void f(poly *r)
{
  __ESBMC_requires(__ESBMC_is_fresh(r, sizeof(poly)));
  __ESBMC_assigns(r->coeffs);
  unsigned i;
  __ESBMC_loop_invariant(
    i <= N && __ESBMC_forall(
                &j,
                !(j >= i && j < N) ||
                  r->coeffs[j] == __ESBMC_old(*r).coeffs[j]));
  for (i = 0; i < N; i++)
  {
    r->coeffs[i] = (short)(r->coeffs[i] + 1);
    if (i == 0)
      r->coeffs[N - 1] = 7;
  }
}

int main(void)
{
  poly r;
  f(&r);
  return 0;
}
