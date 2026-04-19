/* mem_array_bounds_fail:
 * Same contract.  Body reads arr[idx+1] instead of arr[idx]:
 * when idx == n-1, arr[idx+1] is one past the end — out-of-bounds.
 * --bounds-check (default) catches this in the enforced body.
 *
 * Expected: VERIFICATION FAILED (array bounds violation)
 */
#include <assert.h>
#include <stddef.h>

int safe_get(const int *arr, int n, int idx)
{
  __ESBMC_requires(arr != NULL);
  __ESBMC_requires(n > 0);
  __ESBMC_requires(idx >= 0 && idx < n);
  __ESBMC_ensures(__ESBMC_return_value == arr[idx]);
  return arr[idx + 1]; /* BUG: off-by-one, overflows when idx == n-1 */
}

int main()
{
  int a[5] = {10, 20, 30, 40, 50};
  int v = safe_get(a, 5, 4); /* idx=4, arr[5] is out of bounds */
  assert(v == 50);
  return 0;
}
