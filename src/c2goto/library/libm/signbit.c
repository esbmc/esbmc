#define __CRT__NO_INLINE /* Don't let mingw insert code */

#include <math.h>

inline int __signbit(double d)
{
__ESBMC_HIDE:;
  return __ESBMC_signd(d);
}

inline int __signbitf(float f)
{
__ESBMC_HIDE:;
  return __ESBMC_signf(f);
}

inline int __signbitl(long double ld)
{
__ESBMC_HIDE:;
  return __ESBMC_signld(ld);
}

inline int _dsign(double d)
{
__ESBMC_HIDE:;
  return __ESBMC_signd(d);
}

inline int _ldsign(long double ld)
{
__ESBMC_HIDE:;
  return __ESBMC_signld(ld);
}

inline int _fdsign(float f)
{
__ESBMC_HIDE:;
  return __ESBMC_signf(f);
}
