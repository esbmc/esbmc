/* Basic12: Boundary value testing
 * Tests contracts with boundary values (0, -1, INT_MAX, etc.)
 */
#include <assert.h>
#include <limits.h>

int increment(int x)
{


  __ESBMC_ensures(__ESBMC_return_value > x);

  return x + 1;
}

int decrement(int x)
{


  __ESBMC_ensures(__ESBMC_return_value < x);

  return x - 1;
}

int negate(int x)
{
  __ESBMC_ensures((x >= 0 && __ESBMC_return_value <= 0) || (x < 0 && __ESBMC_return_value > 0));
  return -x;
}

int main()
{



  // Test with zero
  int r1 = increment(0);
  assert(r1 == 1);
  
  // Test with negative
  int r2 = decrement(-5);
  assert(r2 == -6);
  
  // Test with positive
  int r3 = negate(10);
  assert(r3 == -10);
  
  return 0;
}

