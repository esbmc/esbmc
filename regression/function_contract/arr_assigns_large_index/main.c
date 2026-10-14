/*
 * arr_assigns_large_index:
 *   Phase 2B checks that arr[idx] is the only element written by ranging a
 *   nondet witness j over the array.  The witness bound is derived from the
 *   extent the contract states, so a statically-fixed index of 150 is in
 *   range as long as __ESBMC_is_fresh covers it.  This bound used to be a
 *   constant 100 elements, which made any idx >= 100 a false ASSERTION
 *   FAILED regardless of the contract.
 */
int write_large_index(int *arr, int v)
{
  __ESBMC_requires(arr != (int *)0);
  __ESBMC_requires(__ESBMC_is_fresh(arr, 200 * sizeof(int)));
  __ESBMC_assigns(arr[150]);
  __ESBMC_ensures(__ESBMC_return_value == 0);
  arr[150] = v;
  return 0;
}

int main()
{
  return 0;
}
