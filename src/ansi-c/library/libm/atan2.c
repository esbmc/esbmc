#define __CRT__NO_INLINE /* Don't let mingw insert code */

#include <math.h>
#include "../intrinsics.h"

#undef atan2

static double _atan(double f, int n)
{
  double p0 = -0.136887688941919269e2;
  double p1 = -0.205058551958616520e2;
  double p2 = -0.849462403513206835e1;
  double p3 = -0.837582993681500593e0;
  double q0 = 0.410663066825757813e2;
  double q1 = 0.861573495971302425e2;
  double q2 = 0.595784361425973445e2;
  double q3 = 0.150240011600285761e2;
  double root_eps = 0.372529029846191406e-8;        /* 2**-(t/2), t = 56    */

  double a[] = {0.0, M_PI/6, M_PI_2, M_PI/3};

  double g, q, r;

  if(f > (2 - sqrt(3)))
  {
    f = ((((sqrt(3) - 1) * f - 0.5) - 0.5) + f) / (sqrt(3) + f);
    n++;
  }
  if(f > root_eps || f < -root_eps)
  {
    g = f * f;
    q = (((g + q3) * g + q2) * g + q1) * g + q0;
    r = (((p3 * g + p2) * g + p1) * g + p0) * g / q;
    f = f + f * r;
  }
  if(n > 1)
    f = -f;
  return (f + a[n]);
}

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

