// As github_4219_old_in_forall_ptr_to_array, but the body adds 2, which the
// ensures clause rejects.
#define N 4
#define BOUND 100

void bump(int (*r)[N])
{
  unsigned j;
  __ESBMC_requires(__ESBMC_is_fresh(r, sizeof(int[N])));
  __ESBMC_requires(
    __ESBMC_forall(&j, !(j < N) || ((*r)[j] > -BOUND && (*r)[j] < BOUND)));
  __ESBMC_ensures(
    __ESBMC_forall(&j, !(j < N) || ((*r)[j] == __ESBMC_old((*r)[j]) + 1)));
  __ESBMC_assigns(*r);

  for (unsigned i = 0; i < N; i++)
    (*r)[i] = (*r)[i] + 2;
}

int main(void)
{
  return 0;
}
