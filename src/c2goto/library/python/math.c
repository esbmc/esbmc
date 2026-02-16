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

  if (x == 1.0)
    return 0.0;
  return log(x);
}

double __ESBMC_acos(double x)
{
__ESBMC_HIDE:;
  return acos(x);
}

double __ESBMC_atan(double x)
{
__ESBMC_HIDE:;
  return atan(x);
}

double __ESBMC_atan2(double y, double x)
{
__ESBMC_HIDE:;
  return atan2(y, x);
}

double __ESBMC_log2(double x)
{
__ESBMC_HIDE:;
  __ESBMC_assert(x > 0.0, "math domain error");
  return log2(x);
}

double __ESBMC_pow(double x, double y)
{
__ESBMC_HIDE:;
  return pow(x, y);
}

double __ESBMC_fabs(double x)
{
__ESBMC_HIDE:;
  return fabs(x);
}

double __ESBMC_trunc(double x)
{
__ESBMC_HIDE:;
  return trunc(x);
}

double __ESBMC_fmod(double x, double y)
{
__ESBMC_HIDE:;
  return fmod(x, y);
}

double __ESBMC_copysign(double x, double y)
{
__ESBMC_HIDE:;
  return copysign(x, y);
}

double __ESBMC_tan(double x)
{
__ESBMC_HIDE:;
  /* Python math.tan wrapper */
  return tan(x);
}

double __ESBMC_asin(double x)
{
__ESBMC_HIDE:;
  /* Python math.asin wrapper */
  return asin(x);
}

double __ESBMC_sinh(double x)
{
__ESBMC_HIDE:;
  /* Python math.sinh wrapper */
  return sinh(x);
}

double __ESBMC_cosh(double x)
{
__ESBMC_HIDE:;
  /* Python math.cosh wrapper */
  return cosh(x);
}

double __ESBMC_tanh(double x)
{
__ESBMC_HIDE:;
  /* Python math.tanh wrapper */
  return tanh(x);
}

double __ESBMC_log10(double x)
{
__ESBMC_HIDE:;
  /* Python math.log10 wrapper */
  __ESBMC_assert(x > 0.0, "math domain error");
  return log10(x);
}
