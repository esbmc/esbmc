/* Basic16: All comparison operators in ensures
 * Comprehensive test of all comparison operators: >, <, >=, <=, ==, !=
 */
#include <assert.h>

int is_positive(int x)
{


  __ESBMC_ensures(__ESBMC_return_value == 0 || __ESBMC_return_value == 1);

  return (x > 0) ? 1 : 0;
}

int is_negative(int x)
{


  __ESBMC_ensures(__ESBMC_return_value == 0 || __ESBMC_return_value == 1);

  return (x < 0) ? 1 : 0;
}

int compare(int a, int b)
{


  __ESBMC_ensures(__ESBMC_return_value >= -1);
  __ESBMC_ensures(__ESBMC_return_value <= 1);
  __ESBMC_ensures(__ESBMC_return_value == -1 || __ESBMC_return_value == 0 || __ESBMC_return_value == 1);

  if (a < b)
    return -1;
  if (a > b)
    return 1;
  return 0;
}

int not_equal(int a, int b)
{


  __ESBMC_ensures(__ESBMC_return_value == 0 || __ESBMC_return_value == 1);

  if (a == b)
    return 0;
  return 1;
}

int main()
{



  int r1 = is_positive(5);
  assert(r1 > 0 || r1 == 0);
  
  int r2 = is_negative(-5);
  assert(r2 == 0 || r2 == 1);
  
  int r3 = compare(5, 10);
  assert(r3 >= -1 && r3 <= 1);
  
  int r4 = not_equal(5, 10);
  assert(r4 != 0 || 5 != 10);
  
  return 0;
}

