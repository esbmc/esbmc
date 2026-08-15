#define N 4
#define BOUND 100

typedef struct
{
  int coeffs[N];
} poly;

void add(poly *r, const poly *b)
{
  unsigned j;
  __ESBMC_requires(__ESBMC_is_fresh(r, sizeof(poly)));
  __ESBMC_requires(__ESBMC_is_fresh(b, sizeof(poly)));
  __ESBMC_requires(__ESBMC_forall(
    &j, !(j < N) || (r->coeffs[j] > -BOUND && r->coeffs[j] < BOUND)));
  __ESBMC_requires(__ESBMC_forall(
    &j, !(j < N) || (b->coeffs[j] > -BOUND && b->coeffs[j] < BOUND)));
  __ESBMC_ensures(__ESBMC_forall(
    &j, !(j < N) || (r->coeffs[j] == __ESBMC_old(r->coeffs[j]) + b->coeffs[j])));
  __ESBMC_assigns(r->coeffs);

  for (unsigned i = 0; i < N; i++)
    r->coeffs[i] = r->coeffs[i] + b->coeffs[i];
}

int main(void)
{
  return 0;
}
