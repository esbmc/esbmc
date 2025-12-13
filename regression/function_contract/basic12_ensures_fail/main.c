/* Basic12_ensures_fail: Boundary value violation
 * This should FAIL - violates ensures with boundary value
 */
#include <assert.h>

int increment(int x)
{


  __ESBMC_ensures(__ESBMC_return_value > x);

  // VIOLATION: returns same value (not greater)
  return x;  // Should be x + 1 (greater than x)
}

int main()
{



  int result = increment(5);
  return 0;
}

