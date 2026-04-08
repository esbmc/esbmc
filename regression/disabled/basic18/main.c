/* Basic18: --function option test
 * Tests that --function option works correctly with contract enforcement
 * This test should be run with: --function test_main --enforce-contract "*"
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

int main()
{



  // This main should not be used when --function test_main is specified
  return 0;
}

