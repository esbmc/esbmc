/* Test 5: __ESBMC_return_value replacement in replace mode
 * Expected: VERIFICATION SUCCESSFUL
 * Tests that __ESBMC_return_value in ensures is replaced with actual return value
 */
#include <assert.h>

int square(int x)
{
  __ESBMC_requires(x >= 0);
  __ESBMC_ensures(__ESBMC_return_value == x * x);
  return x * x;
}

int main()
{
  int a = 5;
  int result = square(a);  // Call replaced with contract
  
  // After contract replacement:
  // - requires: assert(x >= 0) - should pass
  // - ensures: assume(__ESBMC_return_value == x * x)
  //   where __ESBMC_return_value should be replaced with 'result'
  // So: assume(result == a * a) = assume(result == 25)
  assert(result == 25);  // Should hold due to ensures assumption
  
  return 0;
}

