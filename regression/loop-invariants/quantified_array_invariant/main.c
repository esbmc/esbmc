/* Quantified postcondition over an array filled by the loop (GitHub #6217).
 * __ESBMC_forall(&i, !(i < N) || ...) needs i == N at exit, so it is exit
 * reasoning and wants --loop-invariant-check.  The loop is cut rather than
 * unrolled, so --unwind only has to cover the rest of the function and the
 * cost is independent of N. */
#include <stdint.h>

#define N 1024
#define Q 3329

typedef struct
{
  int16_t e[N];
} poly;

static int16_t reduce(int16_t c)
{
  __ESBMC_requires(c > -Q && c < Q);
  __ESBMC_ensures(__ESBMC_return_value >= 0 && __ESBMC_return_value < Q);

  if (c < 0)
    c = (int16_t)(c + Q);
  return c;
}

void normalise(poly *p)
{
  unsigned i, j;
  __ESBMC_requires(__ESBMC_is_fresh(p, sizeof(poly)));
  __ESBMC_ensures(__ESBMC_forall(&i, !(i < N) || (p->e[i] >= 0 && p->e[i] < Q)));

  __ESBMC_loop_invariant(
    i <= N && __ESBMC_forall(&j, !(j < i) || (p->e[j] >= 0 && p->e[j] < Q)));
  for (i = 0; i < N; i++)
  {
    int16_t c = p->e[i];
    __ESBMC_assume(c > -Q && c < Q);
    p->e[i] = reduce(c);
  }
}

int main(void)
{
  poly p;
  normalise(&p);
  return 0;
}
