
#include <math.h>

double acos(double x)
{
__ESBMC_HIDE:;
  return atan2(sqrt(1.0 - x * x), x);
}

double arccos(double x)
{
__ESBMC_HIDE:;
  return acos(x);
}

double __acos(double x)
{
__ESBMC_HIDE:;
  return acos(x);
}

// This function is used by the Python frontend
void __arccos_array(const double *v, double *out, int size)
{
__ESBMC_HIDE:;
  int i = 0;
  while (i < size)
  {
    out[i] = arccos(v[i]);
    ++i;
  }
}
