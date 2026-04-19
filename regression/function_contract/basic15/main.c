/* Basic15: Large numbers and edge cases
 * Tests contracts with large integer values
 */
#include <assert.h>

int add_large(int a, int b)
{


  __ESBMC_ensures(__ESBMC_return_value == a + b);

  return a + b;
}

int multiply(int a, int b)
{
  __ESBMC_ensures(__ESBMC_return_value >= 0 || (a < 0 && b < 0));
  return a * b;
}

int main()
{



  int r1 = add_large(1000, 2000);
  assert(r1 == 3000);
  
  int r2 = multiply(100, 50);
  assert(r2 == 5000);
  
  int r3 = add_large(-1000, 500);
  assert(r3 == -500);
  
  return 0;
}

