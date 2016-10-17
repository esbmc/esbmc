#define __CRT__NO_INLINE /* Don't let mingw insert code */

#include <math.h>
#include "../intrinsics.h"

#undef atan2

double atan2(double v, double u)
{
  double au, av, f;

  av = v < 0.0 ? -v : v;
  au = u < 0.0 ? -u : u;
  if(u != 0.0)
  {
    if(av > au)
    {
      if((f = au / av) == 0.0)
        f = M_PI_2;
      else
        f = _atan(f, 2);
    }
    else
    {
      if((f = av / au) == 0.0)
        f = 0.0;
      else
        f = _atan(f, 0);
    }
  }
  else
  {
    if(v != 0)
      f = M_PI_2;
    else
    {
      f = 0.0;
    }
  }
  if(u < 0.0)
    f = M_PI - f;
  return (v < 0.0 ? -f : f);
}

