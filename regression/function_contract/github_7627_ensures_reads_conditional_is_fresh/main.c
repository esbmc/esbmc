// #7627: a region snapshot that ensures reads is still built, so its extent
// must still come from an unconditional __ESBMC_is_fresh.
#define N 5

void bump(int r[N], int c)
{
  unsigned k;
  __ESBMC_requires(c > 0 ? __ESBMC_is_fresh(r, N * sizeof(int)) : 1);
  __ESBMC_ensures(
    __ESBMC_forall(&k, !(k < N) || r[k] == __ESBMC_old(r[k]) + c));
  __ESBMC_assigns(r[0], r[1], r[2], r[3], r[4]);

  for (unsigned i = 0; i < N; i++)
    r[i] = r[i] + c;
}

int main(void)
{
  int arr[N] = {1, 2, 3, 4, 5};
  bump(arr, 2);
  return 0;
}
