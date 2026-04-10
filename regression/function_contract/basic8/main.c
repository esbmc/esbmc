/* Basic8: Zero and boundary values
 * Tests contracts with zero and boundary conditions
 */
#include <assert.h>

int is_non_negative(int x)
{


  __ESBMC_ensures(__ESBMC_return_value >= 0);

  return (x >= 0) ? 1 : 0;
}

int abs(int x)
{


  __ESBMC_ensures(__ESBMC_return_value >= 0);

  if (x < 0)
    return -x;
  return x;
}

int main()
{



  int result1 = is_non_negative(0);
  assert(result1 >= 0);
  
  int result2 = abs(-5);
  assert(result2 >= 0);
  assert(result2 == 5);
  
  int result3 = abs(5);
  assert(result3 >= 0);
  assert(result3 == 5);
  
  return 0;
}

