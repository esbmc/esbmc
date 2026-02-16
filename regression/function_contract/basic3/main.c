/* Example 2: Equality and inequality operators
 * Tests == and != operators in contracts
 */
#include <assert.h>

int abs_value(int x)
{
  __ESBMC_ensures(__ESBMC_return_value >= 0);
  __ESBMC_ensures((x >= 0 && __ESBMC_return_value == x) || (x < 0 && __ESBMC_return_value == 0 - x));
  if (x < 0)
    return -x;
  return x;
}

int is_zero(int x)
{
  __ESBMC_ensures(__ESBMC_return_value == 0 || __ESBMC_return_value == 1);
  return (x == 0) ? 1 : 0;
}

int main()
{



  int a = -5;
  int result1 = abs_value(a);
  assert(result1 >= 0);
  assert(result1 == 5);
  
  int result2 = is_zero(0);
  assert(result2 == 1);
  
  return 0;
}

