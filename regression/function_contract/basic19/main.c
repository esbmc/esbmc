/* Basic19: --function option with specific contract enforcement
 * Tests that --function works with --enforce-contract for specific functions
 * This test should be run with: --function test_main --enforce-contract add
 */
#include <assert.h>

int add(int x, int y)
{


  __ESBMC_ensures(__ESBMC_return_value == x + y);

  return x + y;
}

int multiply(int x, int y)
{


  __ESBMC_ensures(__ESBMC_return_value == x * y);

  // This function's contract should NOT be enforced (only add is specified)
  return x * y;
}

int test_main()
{



  int r1 = add(5, 10);
  assert(r1 == 15);
  
  int r2 = multiply(3, 4);
  assert(r2 == 12);
  
  return 0;
}

