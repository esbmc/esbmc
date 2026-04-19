/* Basic13_ensures_fail: Multiple ensures violation
 * This should FAIL - violates one of the ensures clauses
 */
#include <assert.h>

int square(int x)
{


  __ESBMC_ensures(__ESBMC_return_value >= 0);
  __ESBMC_ensures(__ESBMC_return_value >= x || x < 0);

  // VIOLATION: returns negative value (violates first ensures)
  return -(x * x);
}

int main()
{



  int result = square(5);
  return 0;
}

