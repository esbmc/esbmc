/* mem_array_bounds_pass:
 * Contract uses requires(0 <= idx && idx < n) to express that the index
 * must be within bounds.  The body then accesses arr[idx] safely.
 * Caller passes a valid index; --bounds-check verifies no out-of-bounds read.
 *
 * Expected: VERIFICATION SUCCESSFUL
 */
#include <assert.h>
#include <stddef.h>

int safe_get(const int *arr, int n, int idx)
{
  __ESBMC_requires(arr != NULL);
  __ESBMC_requires(n > 0);
  __ESBMC_requires(idx >= 0 && idx < n);
  __ESBMC_ensures(__ESBMC_return_value == arr[idx]);
  return arr[idx];
}

int main()
{
  int a[5] = {10, 20, 30, 40, 50};
  int v = safe_get(a, 5, 2); /* idx=2 < n=5 — valid */
  assert(v == 30);
  return 0;
}
