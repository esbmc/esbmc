/* Test 4: __ESBMC_old() support in replace mode
 * Expected: VERIFICATION SUCCESSFUL
 * Tests that __ESBMC_old() snapshots are created at call sites
 */
#include <assert.h>
#include <stddef.h>

int add_one(int *x)
{
  __ESBMC_requires(x != NULL);
  __ESBMC_ensures(*x == __ESBMC_old(*x) + 1);
  *x = *x + 1;
  return *x;
}

int main()
{
  int a = 5;
  int result = add_one(&a);  // Call replaced with contract
  
  // After contract replacement:
  // - requires: assert(x != NULL) - should pass
  // - ensures: assume(*x == old(*x) + 1) - means a == old(a) + 1
  // Since ensures is assumed, we can verify it
  assert(a == 6);  // Should hold due to ensures assumption
  
  return 0;
}

