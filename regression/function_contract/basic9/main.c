/* Basic9: All comparison operators
 * Comprehensive test of all comparison operators
 */
#include <assert.h>

int check_greater(int x, int y)
{


  __ESBMC_ensures(__ESBMC_return_value == 1 || __ESBMC_return_value == 0);

  return (x > y) ? 1 : 0;
}

int check_less(int x, int y)
{


  __ESBMC_ensures(__ESBMC_return_value == 1 || __ESBMC_return_value == 0);

  return (x < y) ? 1 : 0;
}

int check_equal(int x, int y)
{


  __ESBMC_ensures(__ESBMC_return_value == 1 || __ESBMC_return_value == 0);

  return (x == y) ? 1 : 0;
}

int check_not_equal(int x, int y)
{


  __ESBMC_ensures(__ESBMC_return_value == 1 || __ESBMC_return_value == 0);

  return (x != y) ? 1 : 0;
}

int main()
{



  assert(check_greater(5, 3) >= 0);
  assert(check_less(3, 5) >= 0);
  assert(check_equal(5, 5) >= 0);
  assert(check_not_equal(5, 3) >= 0);
  
  return 0;
}

