/* Basic11_ensures_fail: Complex logical expression violation
 * This should FAIL - violates ensures clause with && operator
 */
#include <assert.h>

int in_range(int x, int min, int max)
{


  __ESBMC_requires(min <= max);
  __ESBMC_ensures(__ESBMC_return_value >= min && __ESBMC_return_value <= max);

  // VIOLATION: returns value outside range
  return x + 100;  // This violates the ensures clause
}

int main()
{



  int result = in_range(5, 0, 10);
  return 0;
}

