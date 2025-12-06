/* Example 1: Multiple contract clauses
 * Tests multiple requires and ensures clauses
 */
#include <assert.h>

int divide(int x, int y)
{


  __ESBMC_requires(x >= 0);
  __ESBMC_requires(y > 0);
  __ESBMC_ensures(__ESBMC_return_value >= 0);
  __ESBMC_ensures(__ESBMC_return_value <= x);

  return x / y;
}

int main()
{



  int a = 10;
  int b = 3;
  int result = divide(a, b);
  
  assert(result >= 0);
  assert(result <= a);
  
  return 0;
}

