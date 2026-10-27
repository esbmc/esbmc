// Pins the extent find_callsite_is_fresh_extent resolves for a region
// snapshot under --replace-call-with-contract. The forall ensures reads
// __ESBMC_old(r[k]), so the call site builds the region-copy loop, whose exit
// condition names the element count literally (`old_region_i_bump_call_<N> <
// 5`). Forcing the resolved extent to 1 byte flips this bound to `< 1`. A
// snapshot read only from loop_invariant is not built at all (#7627), which
// is why the ensures, not the loop_invariant, carries the region reference.
#define N 5

void bump(int r[N], int c)
{
  unsigned k;
  __ESBMC_requires(__ESBMC_is_fresh(r, N * sizeof(int)));
  __ESBMC_ensures(r[0] == __ESBMC_old(r[0]) + c);
  __ESBMC_ensures(r[1] == __ESBMC_old(r[1]) + c);
  __ESBMC_ensures(r[2] == __ESBMC_old(r[2]) + c);
  __ESBMC_ensures(r[3] == __ESBMC_old(r[3]) + c);
  __ESBMC_ensures(r[4] == __ESBMC_old(r[4]) + c);
  __ESBMC_ensures(
    __ESBMC_forall(&k, !(k < N) || r[k] == __ESBMC_old(r[k]) + c));
  __ESBMC_assigns(r[0], r[1], r[2], r[3], r[4]);

  unsigned i;
  __ESBMC_loop_invariant(
    i <= N &&
    __ESBMC_forall(&k, !(k < i) || (r[k] == __ESBMC_old(r)[k] + c)) &&
    __ESBMC_forall(&k, !(k >= i && k < N) || (r[k] == __ESBMC_old(r)[k])));
  for (i = 0; i < N; i++)
    r[i] = r[i] + c;
}

int main(void)
{
  int arr[N] = {1, 2, 3, 4, 5};
  bump(arr, 2);
  __ESBMC_assert(
    arr[0] == 3 && arr[1] == 4 && arr[2] == 5 && arr[3] == 6 && arr[4] == 7,
    "ensures propagated the correct incremented values to the caller");
  return 0;
}
