/* Basic13: Multiple ensures with different operators
 * Tests multiple ensures clauses with various comparison operators
 */
#include <assert.h>

int square(int x)
{


  __ESBMC_ensures(__ESBMC_return_value >= 0);
  __ESBMC_ensures(__ESBMC_return_value >= x || x < 0);

  return x * x;
}

int abs_diff(int a, int b)
{


  __ESBMC_ensures(__ESBMC_return_value >= 0);
  __ESBMC_ensures(__ESBMC_return_value == a - b || __ESBMC_return_value == b - a);

  if (a > b)
    return a - b;
  return b - a;
}

int main()
{



  int r1 = square(5);
  assert(r1 == 25);
  assert(r1 >= 0);
  
  int r2 = abs_diff(10, 3);
  assert(r2 == 7);
  assert(r2 >= 0);
  
  int r3 = abs_diff(3, 10);
  assert(r3 == 7);
  
  return 0;
}

