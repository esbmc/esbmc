/*
 * arr_assigns_large_index_fail:
 *   Negative companion to arr_assigns_large_index. The assigns clause names
 *   arr[150] only, and the body also writes arr[151], so Phase 2B must find a
 *   witness j == 151 and report the frame violation.
 *
 *   This is the test that pins the witness bound. Its sibling passes whether
 *   or not the bound is satisfiable, so on its own it cannot tell a working
 *   check from one that was discharged vacuously -- which is exactly what an
 *   ASSUME over an empty range used to do to the whole wrapper.
 */
int write_large_index(int *arr, int v)
{
  __ESBMC_requires(__ESBMC_is_fresh(arr, 200 * sizeof(int)));
  __ESBMC_assigns(arr[150]);
  __ESBMC_ensures(__ESBMC_return_value == 0);
  arr[150] = v;
  arr[151] = v; /* not in the assigns clause */
  return 0;
}

int main()
{
  return 0;
}
