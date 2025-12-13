/* Basic10: Large numbers and edge cases
 * Tests contracts with large numbers and edge cases
 */
#include <assert.h>

int add_one(int x)
{


  __ESBMC_ensures(__ESBMC_return_value > x);

  return x + 1;
}

int subtract_one(int x)
{


  __ESBMC_ensures(__ESBMC_return_value < x);

  return x - 1;
}

int main()
{



  int a = 100;
  int b = -100;
  
  int result1 = add_one(a);
  assert(result1 == a + 1);
  
  int result2 = subtract_one(b);
  assert(result2 == b - 1);
  
  return 0;
}

