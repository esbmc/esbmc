#define __CRT__NO_INLINE /* Don't let mingw insert code */

#include <math.h>
#include "../intrinsics.h"

#undef pow

double pow(double base, double exponent)
{
  int result = 1;
  if(exponent == 0)
    return result;
  if(exponent < 0)
    return 1 / pow(base, -exponent);
  float temp = pow(base, exponent / 2);
  if((int) exponent % 2 == 0)
    return temp * temp;
  else
    return (base * temp * temp);
}
