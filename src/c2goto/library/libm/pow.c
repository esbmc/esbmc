#define __CRT__NO_INLINE /* Don't let mingw insert code */

#include <math.h>

double pow(double base, double exponent)
{
__ESBMC_HIDE:;
  double result = 1.0;
  if(exponent == 0)
    return result;

  if(exponent < 0)
  {
    base = 1.0 / base;
    exponent = -exponent;
  }

  while (exponent > 0)
  {
    if((int)exponent % 2 == 0)
    {
      base *= base;
      exponent /= 2;
    }
    else
    {
      result *= base;
      exponent--;
    }
  }
  return result;
}

double __pow(double base, double exponent)
{
__ESBMC_HIDE:;
  return pow(base, exponent);
}
