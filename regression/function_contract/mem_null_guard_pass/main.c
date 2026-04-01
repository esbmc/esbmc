/* mem_null_guard_pass:
 * requires(p != NULL) is the contract-level null guard.
 * --assume-nonnull-valid synthesises a valid malloc'd pointer in the
 * enforcement harness, so the body never sees NULL and
 * the pointer dereference is safe.
 *
 * Expected: VERIFICATION SUCCESSFUL
 */
#include <stddef.h>

void zero_fill(int *p, int n)
{
  __ESBMC_requires(p != NULL);
  __ESBMC_requires(n > 0);
  __ESBMC_ensures(p[0] == 0); /* at least the first element is zeroed */
  for (int i = 0; i < n; i++)
    p[i] = 0;
}

int main() { return 0; }
