/*
 * arr_assigns_large_index_knownbug:
 *   Phase 2B uses a nondet witness j in [0, ARRAY_ALLOC_ELEMS) to verify
 *   that arr[idx] is the only element written.  ARRAY_ALLOC_ELEMS is
 *   currently 100, so any statically-fixed index >= 100 falls outside the
 *   witness range and the check emits a false ASSERTION FAILED.
 *
 *   Known limitation: assigns(arr[idx]) is unsound for idx >= 100.
 *
 *   Expected (correct): VERIFICATION SUCCESSFUL
 *   Current (bug):      VERIFICATION FAILED — false positive
 */
int write_large_index(int *arr, int v)
{
  __ESBMC_requires(arr != (int *)0);
  __ESBMC_assigns(arr[150]);
  __ESBMC_ensures(__ESBMC_return_value == 0);
  arr[150] = v;
  return 0;
}

int main()
{
  return 0;
}
