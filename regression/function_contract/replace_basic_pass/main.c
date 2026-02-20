/* Test 1: Basic replace mode - requires satisfied, ensures satisfied
 * Expected: VERIFICATION SUCCESSFUL
 * This tests the basic contract replacement functionality
 */
#include <assert.h>

int increment(int x)
{
  __ESBMC_requires(x > 0);
  __ESBMC_ensures(__ESBMC_return_value > x);
  return x + 1;
}

int main()
{
  int a = 5;  // Satisfies requires (x > 0)
  int result = increment(a);  // Call replaced with contract
  
  // Contract ensures result > a, so this should always hold
  assert(result > a);
  
  return 0;
}

