#include <assert.h>
#include <float.h>
#include <limits.h>
#include <math.h>

double nondet_double();

int main()
{
  assert(ilogb(8.0) == 3);
  assert(ilogbf(0.75f) == -1);
  assert(ilogbl(1024.0L) == 10);
  assert(ilogb(DBL_TRUE_MIN) == -1074);
  assert(ilogb(0.0) == FP_ILOGB0);
  assert(ilogb(-INFINITY) == INT_MAX);
  assert(logb(-12.0) == 3.0);
  assert(logbf(0.1f) == -4.0f);
  assert(logb(0.0) == -INFINITY);
  assert(logb(-INFINITY) == INFINITY);

  double x = nondet_double();
  __ESBMC_assume(x >= 1.0 && x < 2.0);
  assert(ilogb(x) == 0);
  assert(logb(-x) == 0.0);
  return 0;
}
