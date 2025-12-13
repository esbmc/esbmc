/* Basic14: Complex nested logical expressions
 * Tests deeply nested || and && operators
 */
#include <assert.h>

int classify(int x)
{


  __ESBMC_ensures(__ESBMC_return_value == 0 || __ESBMC_return_value == 1 || __ESBMC_return_value == 2);

  if (x < 0)
    return 0;
  if (x == 0)
    return 1;
  return 2;
}

int in_interval(int x, int a, int b, int c, int d)
{
  __ESBMC_requires(a <= b);
  __ESBMC_requires(c <= d);
  __ESBMC_ensures((__ESBMC_return_value >= a && __ESBMC_return_value <= b) || (__ESBMC_return_value >= c && __ESBMC_return_value <= d));
  if (x >= a && x <= b)
    return x;
  if (x >= c && x <= d)
    return x;
  if (x < a)
    return a;
  return d;
}

int main()
{



  int r1 = classify(-5);
  assert(r1 == 0 || r1 == 1 || r1 == 2);
  
  int r2 = in_interval(5, 0, 10, 20, 30);
  assert((r2 >= 0 && r2 <= 10) || (r2 >= 20 && r2 <= 30));
  
  return 0;
}

