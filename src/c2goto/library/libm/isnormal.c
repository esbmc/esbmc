#define __CRT__NO_INLINE /* Don't let mingw insert code */

#include <math.h>

inline int __isnormalf(float f)
{
__ESBMC_HIDE:;
  return __ESBMC_isnormalf(f);
}

inline int __isnormald(double d)
{
__ESBMC_HIDE:;
  return __ESBMC_isnormald(d);
}

inline int __isnormall(long double ld)
{
__ESBMC_HIDE:;
  return __ESBMC_isnormalld(ld);
}
