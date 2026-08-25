// __ESBMC_old(r[j]) needs the region's byte extent, which the contracts
// pass reads from an unconditional, direct __ESBMC_is_fresh(r, N) clause.
// With none stated at all, this must hit a specific diagnostic rather than
// silently materialize an unbounded or garbage-sized snapshot (#7057).
#define N 4

void nofresh(int *r)
{
  unsigned j;
  __ESBMC_ensures(__ESBMC_forall(&j, !(j < N) || (r[j] == __ESBMC_old(r[j]) + 1)));
  __ESBMC_assigns(r);

  for (unsigned i = 0; i < N; i++)
    r[i] = r[i] + 1;
}

int main(void)
{
  return 0;
}
