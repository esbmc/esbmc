#define __CRT__NO_INLINE /* Don't let mingw insert code */

#include <math.h>
#include "../intrinsics.h"

#undef sqrt

/*Returns the square root of n. Note that the function */
/*Babylonian method*/
/*http://www.geeksforgeeks.org/square-root-of-a-perfect-square/*/
double babylonian_sqrt(double n)
{
  /*We are using n itself as initial approximation
   This can definitely be improved */
  double x = n;
  double y = 1;
  double e = 1;
  int i = 0;
  while(i++ < 15) //Change this line to increase precision
  {
    x = (x + y) / 2.0;
    y = n / x;
  }
  return x;
}

float sqrtf(float n)
{
  __ESBMC_HIDE:;

  // If not running in floatbv mode, using our old method
  if(!__ESBMC_floatbv_mode())
    return babylonian_sqrt(n);

  // If n is NaN, return NaN
  if(__ESBMC_isnanf(n))
    return NAN;

  // If n == +/-0.0, return +/- 0.0
  if(n == 0.0)
    return n;

  // if n < 0.0, return NAN
  if(n < 0.0)
    return NAN;

  // if +inf, return +inf
  if(__ESBMC_isinff(n))
    return INFINITY;

  return __ESBMC_sqrtf(n);
}

double sqrt(double n)
{
  __ESBMC_HIDE:;

  // If not running in floatbv mode, using our old method
  if(!__ESBMC_floatbv_mode())
    return babylonian_sqrt(n);

  // If n is NaN, return NaN
  if(__ESBMC_isnand(n))
    return NAN;

  // If n == +/-0.0, return +/- 0.0
  if(n == 0.0)
    return n;

  // if n < 0.0, return NAN
  if(n < 0.0)
    return NAN;

  // if +inf, return +inf
  if(__ESBMC_isinfd(n))
    return INFINITY;

  return __ESBMC_sqrtd(n);
}

long double sqrtl(long double n)
{
  __ESBMC_HIDE:;

  // If not running in floatbv mode, using our old method
  if(!__ESBMC_floatbv_mode())
    return babylonian_sqrt(n);

  // If n is NaN, return NaN
  if(__ESBMC_isnanld(n))
    return NAN;

  // If n == +/-0.0, return +/- 0.0
  if(n == 0.0)
    return n;

  // if n < 0.0, return NAN
  if(n < 0.0)
    return NAN;

  // if +inf, return +inf
  if(__ESBMC_isinfld(n))
    return INFINITY;

  return __ESBMC_sqrtld(n);
}
