#include <math.h>

int main()
{
  double x = nondet_double();
  double y = nondet_double();

  __ESBMC_assert(isless(x, y) == (x < y), "isless agrees with <");
  __ESBMC_assert(isgreater(x, y) == (x > y), "isgreater agrees with >");
  __ESBMC_assert(islessequal(x, y) == (x <= y), "islessequal agrees with <=");
  __ESBMC_assert(
    isgreaterequal(x, y) == (x >= y), "isgreaterequal agrees with >=");
  __ESBMC_assert(
    islessgreater(x, y) == (x < y || x > y),
    "islessgreater agrees with < or >");
  __ESBMC_assert(
    isunordered(x, y) == !(x < y || x > y || x == y),
    "isunordered is the absence of an ordering");

  __ESBMC_assert(
    isless(x, y) + isgreater(x, y) + (x == y) + isunordered(x, y) == 1,
    "exactly one ordering relation holds");

  return 0;
}
