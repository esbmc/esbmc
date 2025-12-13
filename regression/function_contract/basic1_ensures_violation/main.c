/* Test: Ensures clause violation
 * This should FAIL verification because ensures clause is violated
 */
#include <assert.h>

int increment(int x)
{


  __ESBMC_requires(x > 0);
  __ESBMC_ensures(__ESBMC_return_value > x);

  return x;  // VIOLATION: should return x + 1, but returns x (violates ensures)
}

int main()
{



  int a = 5;
  int result = increment(a);
  
  return 0;
}

