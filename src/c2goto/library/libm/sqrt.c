#define __CRT__NO_INLINE /* Don't let mingw insert code */

#include <math.h>

#undef sqrt

/*Returns the square root of n. Note that the function */
/*Babylonian method*/
/*http://www.geeksforgeeks.org/square-root-of-a-perfect-square/*/
double babylonian_sqrt(double n)
{
__ESBMC_HIDE:;
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

#define sqrt_def(type, name, isnan_func, isinf_func, sqrt_func)                \
  type name(type n)                                                            \
  {                                                                            \
  __ESBMC_HIDE:;                                                               \
    /* If not running in floatbv mode, using our old method */                 \
    if(!__ESBMC_floatbv_mode())                                                \
      return babylonian_sqrt(n);                                               \
                                                                               \
    return sqrt_func(n);                                                       \
  }                                                                            \
                                                                               \
  type __##name(type f)                                                        \
  {                                                                            \
  __ESBMC_HIDE:;                                                               \
    return name(f);                                                            \
  }

sqrt_def(float, sqrtf, isnanf, isinff, __ESBMC_sqrtf);
sqrt_def(double, sqrt, isnan, isinf, __ESBMC_sqrtd);
sqrt_def(long double, sqrtl, isnanl, isinfl, __ESBMC_sqrtld);

#undef sqrt_def
