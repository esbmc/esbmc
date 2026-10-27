// #7627: __ESBMC_old(r)[k] read only from loop_invariant is never read under
// --replace-call-with-contract, so its extent is not resolved;
// a conditional is_fresh no longer aborts.
#define N 5

void bump(int r[N], int c)
{
  unsigned k;
  __ESBMC_requires(c > 0 ? __ESBMC_is_fresh(r, N * sizeof(int)) : 1);
  __ESBMC_ensures(r[0] == __ESBMC_old(r[0]) + c);
  __ESBMC_ensures(r[4] == __ESBMC_old(r[4]) + c);
  __ESBMC_assigns(r[0], r[1], r[2], r[3], r[4]);

  unsigned i;
  __ESBMC_loop_invariant(
    i <= N && __ESBMC_forall(&k, !(k < i) || (r[k] == __ESBMC_old(r)[k] + c)) &&
    __ESBMC_forall(&k, !(k >= i && k < N) || (r[k] == __ESBMC_old(r)[k])));
  for (i = 0; i < N; i++)
    r[i] = r[i] + c;
}

int main(void)
{
  int arr[N] = {1, 2, 3, 4, 5};
  bump(arr, 2);
  __ESBMC_assert(arr[0] == 3 && arr[4] == 7, "ensures reached the caller");

  return 0;
}
