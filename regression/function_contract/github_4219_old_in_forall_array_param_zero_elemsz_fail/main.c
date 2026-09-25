// __ESBMC_old(p[j]) needs to derive an element count from the region's
// byte extent, dividing by the element type's byte size. A zero-size
// element type (an empty struct -- a real, if unusual, C construct) would
// divide by zero; this must hit a specific diagnostic instead (#7057).
struct Empty
{
};

void bump(struct Empty *p, unsigned n)
{
  unsigned j;
  __ESBMC_requires(__ESBMC_is_fresh(p, n * sizeof(struct Empty)));
  __ESBMC_ensures(
    __ESBMC_forall(&j, !(j < n) || (&p[j] == &__ESBMC_old(p[j]))));
  __ESBMC_assigns();
}

int main(void)
{
  return 0;
}
