#include <complex.h>
#include <math.h>

double creal(double complex z)
{
__ESBMC_HIDE:;
  return __real__ z;
}

double cimag(double complex z)
{
__ESBMC_HIDE:;
  return __imag__ z;
}

double cabs(double complex z)
{
__ESBMC_HIDE:;
  return sqrt(__real__ z * __real__ z + __imag__ z * __imag__ z);
}

double carg(double complex z)
{
__ESBMC_HIDE:;
  return atan2(__imag__ z, __real__ z);
}
