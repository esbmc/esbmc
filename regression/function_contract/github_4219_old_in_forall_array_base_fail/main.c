// github_4219_old_in_forall_array_base with an implementation that adds two,
// so the snapshot the lift produces is the thing being compared against.
#define N 4
#define BOUND 100

int g[N];

void bump(void)
{
  unsigned j;
  __ESBMC_requires(
    __ESBMC_forall(&j, !(j < N) || (g[j] > -BOUND && g[j] < BOUND)));
  __ESBMC_ensures(
    __ESBMC_forall(&j, !(j < N) || (g[j] == __ESBMC_old(g[j]) + 1)));

  for (unsigned i = 0; i < N; i++)
    g[i] = g[i] + 2;
}

int main(void)
{
  return 0;
}
