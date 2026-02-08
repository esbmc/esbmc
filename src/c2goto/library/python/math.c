#include <math.h>

// Python wrappers
double __ESBMC_sin(double x)
{
__ESBMC_HIDE:;
  return sin(x);
}

double __ESBMC_cos(double x)
{
__ESBMC_HIDE:;
  return cos(x);
}

double __ESBMC_sqrt(double x)
{
__ESBMC_HIDE:;
  return sqrt(x);
}

double __ESBMC_exp(double x)
{
__ESBMC_HIDE:;
  return exp(x);
}

double __ESBMC_log(double x)
{
__ESBMC_HIDE:;
  __ESBMC_assert(x > 0.0, "math domain error");

  if (x == 1.0) return 0.0;
  return log(x);
}