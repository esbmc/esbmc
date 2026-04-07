/* mem_null_guard_fail:
 * Body dereferences p without checking NULL first.
 * Contract says requires(p != NULL).
 * Caller passes NULL, so the requires assertion fires.
 *
 * Expected: VERIFICATION FAILED (contract requires violated)
 */
#include <stddef.h>
#include <assert.h>

int deref(int *p)
{
  __ESBMC_requires(p != NULL);
  __ESBMC_ensures(__ESBMC_return_value == *p);
  return *p;
}

int main()
{
  int *q = NULL;
  int v = deref(q); /* passes NULL — violates requires */
  return 0;
}
