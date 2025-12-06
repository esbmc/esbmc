/* Basic23: --function isolation test (should fail without --function)
 * Tests that without --function, main() is used and violations are detected
 * This test should be run with: --enforce-contract "*" (no --function)
 * The main() function should be used, and wrong_function's violation should be detected
 */
#include <assert.h>

int wrong_function(int x)
{


  __ESBMC_ensures(__ESBMC_return_value == x + 1);

  // This function violates its contract
  return x + 2;  // VIOLATION: should return x + 1
}

int main()
{



  // Without --function, main() is used, so wrong_function's violation should be detected
  int r = wrong_function(5);
  return 0;
}

