#include <complex.h>
#include <math.h>

#ifdef _MSC_VER
#  define COMPLEX_DOUBLE _Dcomplex
#  define CREAL(z) ((z)._Val[0])
#  define CIMAG(z) ((z)._Val[1])
#else
#  define COMPLEX_DOUBLE double complex
#  define CREAL(z) __real__(z)
#  define CIMAG(z) __imag__(z)
#endif

double creal(COMPLEX_DOUBLE z)
{
__ESBMC_HIDE:;
  return CREAL(z);
}

double cimag(COMPLEX_DOUBLE z)
{
__ESBMC_HIDE:;
  return CIMAG(z);
}

double cabs(COMPLEX_DOUBLE z)
{
__ESBMC_HIDE:;
  return sqrt(CREAL(z) * CREAL(z) + CIMAG(z) * CIMAG(z));
}

double carg(COMPLEX_DOUBLE z)
{
__ESBMC_HIDE:;
  return atan2(CIMAG(z), CREAL(z));
}
