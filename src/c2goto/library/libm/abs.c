#define __CRT__NO_INLINE /* Don't let mingw insert code */

#include <math.h>

int abs(int i)
{
__ESBMC_HIDE:;
  return __ESBMC_abs(i);
}

long labs(long i)
{
__ESBMC_HIDE:;
  return __ESBMC_labs(i);
}

long long llabs(long long i)
{
__ESBMC_HIDE:;
  return __ESBMC_llabs(i);
}
