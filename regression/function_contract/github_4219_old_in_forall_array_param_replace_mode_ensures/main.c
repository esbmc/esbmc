// __ESBMC_old(r[j]) on a pointer parameter, quantified directly in ensures,
// under --replace-call-with-contract (#7057).

#define N 4

void bump(int r[N])
{
  unsigned j;
  __ESBMC_requires(__ESBMC_is_fresh(r, N * sizeof(int)));
  __ESBMC_ensures(__ESBMC_forall(&j, !(j < N) || (r[j] == __ESBMC_old(r[j]) + 1)));
  __ESBMC_assigns(r);

  for (unsigned i = 0; i < N; i++)
    r[i] = r[i] + 1;
}

int main(void)
{
  int arr[N] = {1, 2, 3, 4};
  bump(arr);
  __ESBMC_assert(arr[0] == 2 && arr[3] == 5, "bumped");
  return 0;
}
