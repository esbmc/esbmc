// __ESBMC_old(r[j]) on a pointer parameter needs the extent-aware copy
// loop generate_checking_wrapper builds from param_extents, which
// --replace-call-with-contract's call-site path has no equivalent of --
// there is no allocation to read an extent from at a replaced call site,
// only a valid_object assertion. Rejected with a specific diagnostic
// instead of indexing a pointer-typed (not array-typed) snapshot (#7057).
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
  return 0;
}
