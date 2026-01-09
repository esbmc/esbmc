/* Basic7: Complex expressions with multiple parameters
 * Tests contracts with multiple parameters and complex conditions
 */
#include <assert.h>

int max(int a, int b)
{


  __ESBMC_ensures(__ESBMC_return_value >= a);
  __ESBMC_ensures(__ESBMC_return_value >= b);

  return (a > b) ? a : b;
}

int min(int a, int b)
{


  __ESBMC_ensures(__ESBMC_return_value <= a);
  __ESBMC_ensures(__ESBMC_return_value <= b);

  return (a < b) ? a : b;
}

int main()
{



  int x = 10;
  int y = 20;
  
  int max_val = max(x, y);
  assert(max_val >= x);
  assert(max_val >= y);
  
  int min_val = min(x, y);
  assert(min_val <= x);
  assert(min_val <= y);
  
  return 0;
}

