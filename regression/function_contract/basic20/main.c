/* Basic20: Without --function option (default main)
 * Tests that contract enforcement works correctly without --function option
 * This test should be run with: --enforce-contract "*" (no --function)
 */
#include <assert.h>

int add(int x, int y)
{


  __ESBMC_ensures(__ESBMC_return_value == x + y);

  return x + y;
}

int subtract(int x, int y)
{


  __ESBMC_ensures(__ESBMC_return_value == x - y);

  return x - y;
}

int main()
{



  int r1 = add(10, 5);
  assert(r1 == 15);
  
  int r2 = subtract(10, 5);
  assert(r2 == 5);
  
  return 0;
}

