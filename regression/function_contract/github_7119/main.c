/* github #7119: the forall guard bounds j_bound on both sides, so it holds. */
typedef short int16_t;
#define N 4
typedef struct
{
  int16_t coeffs[N];
} poly;
int j_bound;

void add(poly *r)
{
  __ESBMC_requires(r != 0);
  __ESBMC_requires(__ESBMC_forall(
    &j_bound,
    !(j_bound >= 0 && j_bound < N) ||
      (r->coeffs[j_bound] > -6658 && r->coeffs[j_bound] < 6658)));
  __ESBMC_ensures(1);
  __ESBMC_assigns(r->coeffs);
  r->coeffs[0] = 0;
}

int main(void)
{
  poly r = {{0, 0, 0, 0}};
  add(&r);
  return 0;
}
