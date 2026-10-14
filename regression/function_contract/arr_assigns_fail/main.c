/* arr_assigns_fail:
 * Function declares __ESBMC_assigns(arr[idx]) but also writes arr[idx+1],
 * which Phase 2B catches with a nondet witness index j: for j != idx, arr[j]
 * must be unchanged.
 *
 * The is_fresh clause is required, not decorative: Phase 2B has to read arr[j]
 * to snapshot it, so it only runs for pointers whose extent the contract
 * states. Without it the witness bound would assume an extent (#6212).
 *
 * Expected: VERIFICATION FAILED
 */
int arr_write2(int *arr, int idx, int val)
{
  __ESBMC_requires(arr != (int *)0 && idx >= 0 && idx < 9);
  __ESBMC_requires(__ESBMC_is_fresh(arr, 10 * sizeof(int)));
  __ESBMC_assigns(arr[idx]);
  __ESBMC_ensures(__ESBMC_return_value == 0);
  arr[idx] = val;
  arr[idx + 1] = val; /* BUG: arr[idx+1] not in assigns */
  return 0;
}

int main() { return 0; }
