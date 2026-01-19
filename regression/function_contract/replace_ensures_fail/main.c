/* Test 3: Ensures violation check in replace mode
 * Expected: VERIFICATION FAILED
 * The ensures clause is assumed, but we assert a condition that contradicts it
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
  int a = 5;  // Satisfies requires
  int result = increment(a);  // Call replaced with contract, ensures is assumed
  
  // This contradicts the ensures clause (result > a)
  // Since ensures is assumed, this assertion should fail
  assert(result <= a);
  
  return 0;
}

