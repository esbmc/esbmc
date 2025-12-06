/* Basic14_ensures_fail: Complex nested logical expression violation
 * This should FAIL - violates ensures with nested || and &&
 */
#include <assert.h>

int in_interval(int x, int a, int b, int c, int d)
{

  __ESBMC_requires(a <= b);
  __ESBMC_requires(c <= d);
  __ESBMC_ensures((__ESBMC_return_value >= a && __ESBMC_return_value <= b) || (__ESBMC_return_value >= c && __ESBMC_return_value <= d));
  // VIOLATION: returns value outside both intervals
  return (a + b + c + d) / 2 + 100;  // Clearly outside both intervals
}

int main()
{



  int result = in_interval(5, 0, 10, 20, 30);
  return 0;
}

