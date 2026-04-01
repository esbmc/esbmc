/* replace_ptr_old_fail:
 * --replace-call-with-contract with pointer param + __ESBMC_old().
 * The ensures says *p == old(*p) + 1, so after replacement x == 42.
 * The assertion x == 43 is wrong and must be caught.
 *
 * Before the pointer-param havoc fix this returned VERIFICATION SUCCESSFUL
 * because ASSUME(41==42) = ASSUME(FALSE) made every assertion vacuously true.
 * With the fix the verifier correctly reports VERIFICATION FAILED.
 *
 * Expected: VERIFICATION FAILED
 */
#include <assert.h>
#include <stddef.h>

void increment(int *p)
{
  __ESBMC_requires(p != NULL);
  __ESBMC_ensures(*p == __ESBMC_old(*p) + 1);
  (*p)++;
}

int main()
{
  int x = 41;
  increment(&x);
  assert(x == 43); /* wrong: ensures only guarantees x == 42 */
  return 0;
}
