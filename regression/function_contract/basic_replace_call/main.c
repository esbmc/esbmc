/* Test: Requires violation with --replace-call-with-contract
 * This should FAIL verification because requires clause is violated at call site
 * Note: This test is for --replace-call-with-contract (system-level verification)
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



  int a = -5;  // VIOLATION: violates requires clause (x > 0)
  int result = increment(a);  // This call should be replaced with contract check
  
  return 0;
}

