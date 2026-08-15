// An `int r[N]` parameter is a pointer, so there is no whole array to snapshot
// and the lift declines. The quantified __ESBMC_old then reaches the frontend
// unchanged and reports "cannot model call to __ESBMC_old_raw on a quantified
// variable". Wrapping the array in a struct (github_4219_old_in_forall) or
// naming it directly (github_4219_old_in_forall_array_base) both work.
#define N 4
#define BOUND 100

void bump(int r[N])
{
  unsigned j;
  __ESBMC_requires(__ESBMC_is_fresh(r, N * sizeof(int)));
  __ESBMC_requires(
    __ESBMC_forall(&j, !(j < N) || (r[j] > -BOUND && r[j] < BOUND)));
  __ESBMC_ensures(
    __ESBMC_forall(&j, !(j < N) || (r[j] == __ESBMC_old(r[j]) + 1)));
  __ESBMC_assigns(r);

  for (unsigned i = 0; i < N; i++)
    r[i] = r[i] + 1;
}

int main(void)
{
  return 0;
}
