/* Basic22: --function isolation test
 * Tests that --function correctly isolates main() function
 * This test should be run with: --function test_main --enforce-contract "*"
 * The main() function should be completely ignored
 */
#include <assert.h>

int correct_function(int x)
{


  __ESBMC_ensures(__ESBMC_return_value == x + 1);

  return x + 1;
}

int wrong_function(int x)
{


  __ESBMC_ensures(__ESBMC_return_value == x + 1);

  // This function violates its contract, but should NOT be checked
  // because it's only called from main(), which is ignored
  return x + 2;  // VIOLATION: should return x + 1
}

int test_main()
{



  // Only this function should be verified
  int r = correct_function(5);
  assert(r == 6);
  return 0;
}

int main()
{



  // This function should be completely ignored when --function test_main is used
  // Even though wrong_function violates its contract, it should not cause failure
  int r = wrong_function(5);
  assert(r == 7);  // This assertion would fail if wrong_function returned x+1
  return 0;
}

