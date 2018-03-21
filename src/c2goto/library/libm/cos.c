#define __CRT__NO_INLINE /* Don't let mingw insert code */

#include <math.h>

double cos(double x)
{
__ESBMC_HIDE:;
  double t, s;
  int p;
  p = 0;
  s = 1.0;
  t = 1.0;
  x = fmod(x + M_PI, M_PI * 2) - M_PI; // restrict x so that -M_PI < x < M_PI
  double xsqr = x * x;
  double ab = 1;
  while((ab > 1e-16) && (p < 15))
  {
    p++;
    t = (-t * xsqr) / (((p << 1) - 1) * (p << 1));
    s += t;
    ab = (s == 0) ? 1 : fabs(t / s);
  }
  return s;
}

double __cos(double x)
{
__ESBMC_HIDE:;
  return cos(x);
}
