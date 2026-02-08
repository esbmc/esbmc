#include <math.h>

// Python wrappers
double __ESBMC_sin(double x)
{
  return sin(x);
}

double __ESBMC_cos(double x)
{
  return cos(x);
}

double __ESBMC_sqrt(double x)
{
  return sqrt(x);
}