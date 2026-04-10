/* Test: Multiple requires clauses - one violated
 * This should FAIL verification because one requires clause is violated
 */
#include <assert.h>

int divide(int x, int y)
{


  __ESBMC_requires(x >= 0);
  __ESBMC_requires(y > 0);
  __ESBMC_ensures(__ESBMC_return_value >= 0);

  return x / y;
}

int main()
{



  int a = 10;
  int b = 0;  // VIOLATION: violates second requires clause (y > 0)
  int result = divide(a, b);
  
  return 0;
}

