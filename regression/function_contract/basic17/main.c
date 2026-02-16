/* Basic17: Complex arithmetic expressions
 * Tests complex arithmetic expressions in contracts: (x + y) * 2, x * x + y * y, etc.
 */
#include <assert.h>

int double_sum(int x, int y)
{
  __ESBMC_ensures(__ESBMC_return_value == (x + y) * 2);
  return (x + y) * 2;
}

int square_sum(int x, int y)
{
  __ESBMC_ensures(__ESBMC_return_value == x * x + y * y);

  return x * x + y * y;
}

int complex_expr(int a, int b, int c)
{


  __ESBMC_ensures(__ESBMC_return_value == a * b + c);
  __ESBMC_ensures(__ESBMC_return_value >= a + b);

  return a * b + c;
}

int main()
{



  int r1 = double_sum(5, 10);
  assert(r1 == 30);
  
  int r2 = square_sum(3, 4);
  assert(r2 == 25);
  
  int r3 = complex_expr(2, 3, 5);
  assert(r3 == 11);
  
  return 0;
}

