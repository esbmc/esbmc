// all classification
#include <math.h>

int main()
{
  double d1, _d1;
  d1=_d1;
  __ESBMC_assume(isnormal(d1));
  assert(!isnan(d1));
  assert(!isinf(d1));
  assert(isfinite(d1));

  double d2, _d2;
  d2=_d2;
  __ESBMC_assume(isinf(d2));
  assert(!isnormal(d2));
  assert(!isnan(d2));

  double d3, _d3;
  d3=_d3;
  __ESBMC_assume(isnan(d3));
  assert(!isnormal(d3));
  assert(!isinf(d3));
  assert(d3!=d3);

  double d4, _d4;
  d4=_d4;
  __ESBMC_assume(isfinite(d4));
  assert(!isnan(d4));
  assert(!isinf(d4));

  double d5, _d5;
  d5=_d5;
  __ESBMC_assume(!isnan(d5) && !isinf(d5));
  assert(isfinite(d5));
}
