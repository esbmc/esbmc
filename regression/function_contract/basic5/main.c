/* Example 4: Negative numbers
 * Tests negative constants in contracts
 */
#include <assert.h>

int is_positive(int x)
{


  __ESBMC_ensures(__ESBMC_return_value > 0 || __ESBMC_return_value == 0);

  return (x > 0) ? 1 : 0;
}

int check_range(int x)
{


  __ESBMC_requires(x >= -10);
  __ESBMC_requires(x <= 10);
  __ESBMC_ensures(__ESBMC_return_value >= -10);
  __ESBMC_ensures(__ESBMC_return_value <= 10);

  return x;
}

int main()
{



  int result1 = is_positive(5);
  assert(result1 >= 0);
  
  int result2 = check_range(-5);
  assert(result2 >= -10);
  assert(result2 <= 10);
  
  return 0;
}

