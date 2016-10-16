#define __CRT__NO_INLINE /* Don't let mingw insert code */
#include <math.h>
#undef fpclassify
#undef isfinite
#undef isfinite
#undef isnormal
#undef isnan
#undef isinf
#undef signbit

#include <fenv.h>

#include "intrinsics.h"

#ifdef _WIN32
#undef fabs
#undef fabsl
#undef fabsf

// Whipped out of glibc headers. Don't exactly know how these work, but they're
// what the linux version works upon.
#ifdef _MSVC
enum
  {
    FP_NAN,
# define FP_NAN FP_NAN
    FP_INFINITE,
# define FP_INFINITE FP_INFINITE
    FP_ZERO,
# define FP_ZERO FP_ZERO
    FP_SUBNORMAL,
# define FP_SUBNORMAL FP_SUBNORMAL
    FP_NORMAL
# define FP_NORMAL FP_NORMAL
  };
#endif
#endif

double fabs(double d)
{
  if(d == 0.0)
    return 0.0;
  if(__ESBMC_isinfd(d))
    return INFINITY;
  if(__ESBMC_isnand(d))
    return NAN;
  return __ESBMC_fabs(d);
}

long double fabsl(long double ld)
{
  if(ld == 0.0)
    return 0.0;
  if(__ESBMC_isinfl(ld))
    return INFINITY;
  if(__ESBMC_isnanl(ld))
    return NAN;
  return __ESBMC_fabsl(ld);
}

float fabsf(float f)
{
  if(f == 0.0)
    return 0.0;
  if(__ESBMC_isinff(f))
    return INFINITY;
  if(__ESBMC_isnanf(f))
    return NAN;
  return __ESBMC_fabsf(f);
}

double fmod(double x, double y)
{
  // If either argument is NaN, NaN is returned
  if(__ESBMC_isnand(x) || __ESBMC_isnand(y))
    return NAN;

  // If x is +inf/-inf and y is not NaN, NaN is returned and FE_INVALID is raised
  if(__ESBMC_isinfd(x))
    return NAN;

  // If y is +0.0/-0.0 and x is not NaN, NaN is returned and FE_INVALID is raised
  if(y == 0.0)
    return NAN;

  // If x is +0.0/-0.0 and y is not zero, +0.0/-0.0 is returned
  if((x == 0.0) && (y != 0.0))
  {
    if(__ESBMC_signd(x))
      return -0.0;
    else
      return +0.0;
  }

  // If y is +inf/-inf and x is finite, x is returned.
  if(__ESBMC_isinfd(y) && __ESBMC_isfinited(x))
    return x;

  return x - (y * (int)(x/y));
}

float fmodf(float x, float y)
{
  // If either argument is NaN, NaN is returned
  if(__ESBMC_isnanf(x) || __ESBMC_isnanf(y))
    return NAN;

  // If x is +inf/-inf and y is not NaN, NaN is returned and FE_INVALID is raised
  if(__ESBMC_isinff(x))
    return NAN;

  // If y is +0.0/-0.0 and x is not NaN, NaN is returned and FE_INVALID is raised
  if(y == 0.0)
    return NAN;

  // If x is +0.0/-0.0 and y is not zero, +0.0/-0.0 is returned
  if((x == 0.0) && (y != 0.0))
  {
    if(__ESBMC_signf(x))
      return -0.0;
    else
      return +0.0;
  }

  // If y is +inf/-inf and x is finite, x is returned.
  if(__ESBMC_isinff(y) && __ESBMC_isfinitef(x))
    return x;

  return x - (y * (int)(x/y));
}

long double fmodl(long double x, long double y)
{
  // If either argument is NaN, NaN is returned
  if(__ESBMC_isnanld(x) || __ESBMC_isnanld(y))
    return NAN;

  // If x is +inf/-inf and y is not NaN, NaN is returned and FE_INVALID is raised
  if(__ESBMC_isinfld(x))
    return NAN;

  // If y is +0.0/-0.0 and x is not NaN, NaN is returned and FE_INVALID is raised
  if(y == 0.0)
    return NAN;

  // If x is +0.0/-0.0 and y is not zero, +0.0/-0.0 is returned
  if((x == 0.0) && (y != 0.0))
  {
    if(__ESBMC_signld(x))
      return -0.0;
    else
      return +0.0;
  }

  // If y is +inf/-inf and x is finite, x is returned.
  if(__ESBMC_isinfld(y) && __ESBMC_isfiniteld(x))
    return x;

  return x - (y * (int)(x/y));
}

float remquof(float x, float y, int *quo)
{
  // If either argument is NaN, NaN is returned
  if(__ESBMC_isnanf(x) || __ESBMC_isnanf(y))
    return NAN;

  // If y is +0.0/-0.0 and x is not NaN, NaN is returned and FE_INVALID is raised
  if(y == 0.0)
    return NAN;

  // If x is +inf/-inf and y is not NaN, NaN is returned and FE_INVALID is raised
  if(__ESBMC_isinff(x))
    return NAN;

  // remainder = x - rquot * y
  // Where rquot is the result of: x/y, rounded toward the nearest
  // integral value (with halfway cases rounded toward the even number).

  // Save previous rounding mode
  int old_rm = fegetround();

  // Set round to nearest
  fesetround(FE_TONEAREST);

  // Perform division
  float rquot = x/y;

  // Restore old rounding mode
  fesetround(old_rm);

  return x - (y * (int)rquot);
}

double remquo(double x, double y, int *quo)
{
  // If either argument is NaN, NaN is returned
  if(__ESBMC_isnand(x) || __ESBMC_isnand(y))
    return NAN;

  // If y is +0.0/-0.0 and x is not NaN, NaN is returned and FE_INVALID is raised
  if(y == 0.0)
    return NAN;

  // If x is +inf/-inf and y is not NaN, NaN is returned and FE_INVALID is raised
  if(__ESBMC_isinfd(x))
    return NAN;

  // remainder = x - rquot * y
  // Where rquot is the result of: x/y, rounded toward the nearest
  // integral value (with halfway cases rounded toward the even number).

  // Save previous rounding mode
  int old_rm = fegetround();

  // Set round to nearest
  fesetround(FE_TONEAREST);

  // Perform division
  double rquot = x/y;

  // Restore old rounding mode
  fesetround(old_rm);

  return x - (y * (int)rquot);
}

long double remquol(long double x, long double y, int *quo)
{
  // If either argument is NaN, NaN is returned
  if(__ESBMC_isnanl(x) || __ESBMC_isnanl(y))
    return NAN;

  // If y is +0.0/-0.0 and x is not NaN, NaN is returned and FE_INVALID is raised
  if(y == 0.0)
    return NAN;

  // If x is +inf/-inf and y is not NaN, NaN is returned and FE_INVALID is raised
  if(__ESBMC_isinfl(x))
    return NAN;

  // remainder = x - rquot * y
  // Where rquot is the result of: x/y, rounded toward the nearest
  // integral value (with halfway cases rounded toward the even number).

  // Save previous rounding mode
  int old_rm = fegetround();

  // Set round to nearest
  fesetround(FE_TONEAREST);

  // Perform division
  long double rquot = x/y;

  // Restore old rounding mode
  fesetround(old_rm);

  return x - (y * (int)rquot);
}

float remainderf(float x, float y)
{
  int quo;
  return remquof(x, y, &quo);
}

double remainder(double x, double y)
{
  int quo;
  return remquo(x, y, &quo);
}

long double remainderl(long double x, long double y)
{
  int quo;
  return remquol(x, y, &quo);
}

float nearbyintf(float f)
{
  if(f == 0.0)
    return f;
  if(__ESBMC_isinff(f))
    return f;
  if(__ESBMC_isnanf(f))
    return NAN;
  return __ESBMC_nearbyintf(f);
}

double nearbyint(double d)
{
  if(d == 0.0)
    return d;
  if(__ESBMC_isinfd(d))
    return d;
  if(__ESBMC_isnand(d))
    return NAN;
  return __ESBMC_nearbyintd(d);
}

long double nearbyintl(long double ld)
{
  if(ld == 0.0)
    return ld;
  if(__ESBMC_isinfl(ld))
    return ld;
  if(__ESBMC_isnanl(ld))
    return NAN;
  return __ESBMC_nearbyintl(ld);
}

int isfinite(double d) { return __ESBMC_isfinited(d); }

int __finite(double d) { return __ESBMC_isfinited(d); }

int __finitef(float f) { return __ESBMC_isfinitef(f); }

int __finitel(long double ld) { return __ESBMC_isfiniteld(ld); }

inline int isinf(double d) { return __ESBMC_isinfd(d); }

inline int __isinf(double d) { return __ESBMC_isinfd(d); }

inline int isinff(float f) { return __ESBMC_isinff(f); }

inline int __isinff(float f) { return __ESBMC_isinff(f); }

inline int isinfl(long double ld) { return __ESBMC_isinfld(ld); }

inline int __isinfl(long double ld) { return __ESBMC_isinfld(ld); }

inline int isnan(double d) { return __ESBMC_isnand(d); }

inline int __isnan(double d) { return __ESBMC_isnand(d); }

inline int __isnanf(float f) { return __ESBMC_isnanf(f); }

inline int isnanf(float f) { return __ESBMC_isnanf(f); }

inline int isnanl(long double ld) { return __ESBMC_isnanld(ld); }

inline int __isnanl(long double ld) { return __ESBMC_isnanld(ld); }

inline int isnormal(double d) { return __ESBMC_isnormald(d); }

inline int __isnormalf(float f) { return __ESBMC_isnormalf(f); }

inline int _dsign(double d) { return __ESBMC_signd(d); }

inline int _ldsign(long double ld) { return __ESBMC_signld(ld); }

inline int _fdsign(float f) { return __ESBMC_signf(f); }

inline int signbit(double d) { return __ESBMC_signd(d); }

inline int __signbitd(double d) { return __ESBMC_signd(d); }

inline int __signbitf(float f) { return __ESBMC_signf(f); }

inline int __signbitl(long double ld) { return __ESBMC_signld(ld); }

inline int __signbit(double d) { return __ESBMC_signd(d); }

int abs(int i) { return __ESBMC_abs(i); }

long int labs(long int i) { return __ESBMC_labs(i); }

inline short _dclass(double d) {
  __ESBMC_HIDE:;
  return __ESBMC_isnand(d)?FP_NAN:
         __ESBMC_isinfd(d)?FP_INFINITE:
         d==0?FP_ZERO:
         __ESBMC_isnormald(d)?FP_NORMAL:
         FP_SUBNORMAL;
}

inline short _ldclass(long double ld) {
  __ESBMC_HIDE:;
  return __ESBMC_isnanld(ld)?FP_NAN:
         __ESBMC_isinfld(ld)?FP_INFINITE:
         ld==0?FP_ZERO:
         __ESBMC_isnormalld(ld)?FP_NORMAL:
         FP_SUBNORMAL;
}

inline short _fdclass(float f) {
  __ESBMC_HIDE:;
  return __ESBMC_isnanf(f)?FP_NAN:
         __ESBMC_isinff(f)?FP_INFINITE:
         f==0?FP_ZERO:
         __ESBMC_isnormalf(f)?FP_NORMAL:
         FP_SUBNORMAL;
}

inline int __fpclassifyd(double d) {
  __ESBMC_HIDE:;
  return __ESBMC_isnand(d)?FP_NAN:
         __ESBMC_isinfd(d)?FP_INFINITE:
         d==0?FP_ZERO:
         __ESBMC_isnormald(d)?FP_NORMAL:
         FP_SUBNORMAL;
}

inline int __fpclassifyl(long double f) {
  __ESBMC_HIDE:;
  return __ESBMC_isnanld(f)?FP_NAN:
         __ESBMC_isinfld(f)?FP_INFINITE:
         f==0?FP_ZERO:
         __ESBMC_isnormalld(f)?FP_NORMAL:
         FP_SUBNORMAL;
}

inline int __fpclassify(double d) {
  __ESBMC_HIDE:;
  return __ESBMC_isnand(d)?FP_NAN:
         __ESBMC_isinfd(d)?FP_INFINITE:
         d==0?FP_ZERO:
         __ESBMC_isnormald(d)?FP_NORMAL:
         FP_SUBNORMAL;
}

inline int __fpclassifyf(float f)
{
  __ESBMC_HIDE:;
  return __ESBMC_isnanf(f)?FP_NAN:
         __ESBMC_isinff(f)?FP_INFINITE:
         f==0?FP_ZERO:
         __ESBMC_isnormalf(f)?FP_NORMAL:
         FP_SUBNORMAL;
}

double cos(double x)
{
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

double sin(double x)
{
  return cos(x - M_PI_2);
}

double acos(double x)
{
  return 1/cos(x);
}

/*Returns the square root of n. Note that the function */
/*Babylonian method*/
/*http://www.geeksforgeeks.org/square-root-of-a-perfect-square/*/
double sqrt(double n)
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

double atan(double x)
{
  double a;

  a = x < 0.0 ? -x : x;
  if(a > 1.0)
    a = _atan(1.0 / a, 2);
  else
    a = _atan(a, 0);
  return (x < 0.0 ? -a : a);
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
