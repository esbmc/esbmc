/* replace_ptr_old_pass:
 * --replace-call-with-contract with pointer param + __ESBMC_old().
 * The ensures says *p == old(*p) + 1.  After replacement the caller
 * correctly knows x == 42 (= 41 + 1) and the assertion x == 42 passes.
 *
 * Without pointer-param havoc this test was vacuously SUCCESSFUL even when
 * the assertion was wrong (x==43) because ASSUME(41==42) collapsed the path.
 * With the fix the ASSUME genuinely constrains the havoced value.
 *
 * Expected: VERIFICATION SUCCESSFUL
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
  assert(x == 42); /* correct: ensures guarantees x == old(x)+1 == 42 */
  return 0;
}
