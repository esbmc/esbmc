/* Basic21: Division and modulo operations
 * Tests division and modulo operations in contracts
 */
#include <assert.h>

int divide(int x, int y)
{


  __ESBMC_ensures(__ESBMC_return_value == x / y);

  return x / y;
}

int remainder(int x, int y)
{
  __ESBMC_ensures(__ESBMC_return_value == x - (x / y) * y);
  return x % y;
}

int main()
{



  int r1 = divide(20, 4);
  assert(r1 == 5);
  
  int r2 = remainder(20, 3);
  assert(r2 == 2);
  
  return 0;
}

