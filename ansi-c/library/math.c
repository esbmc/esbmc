#define __CRT__NO_INLINE /* Don't let mingw insert code */
#include <math.h>
#undef fpclassify
#undef isfinite
#undef isfinite
#undef isnormal
#undef isnan
#undef isinf
#undef signbit

#include "intrinsics.h"

#define M_PI     3.14159265358979323846
#define M_PI_2   1.57079632679489661923132169164      // Pi/2
#define PREC 1e-16
#define M_LN10   2.30258509299404568402
#define DBL_EPSILON 2.2204460492503131e-16

#define M_E      2.71828182845905
#define M_E2    (M_E * M_E)
#define M_E4    (M_E2 * M_E2)
#define M_E8    (M_E4 * M_E4)
#define M_E16   (M_E8 * M_E8)
#define M_E32   (M_E16 * M_E16)
#define M_E64   (M_E32 * M_E32)
#define M_E128  (M_E64 * M_E64)
#define M_E256  (M_E128 * M_E128)
#define M_E512  (M_E256 * M_E256)
#define M_E1024 (M_E512 * M_E512)

//#define INFINITY (1.0 / 0.0)

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

int abs(int i) { return __ESBMC_abs(i); }
long int labs(long int i) { return __ESBMC_labs(i); }
//double fabs(double d) { return __ESBMC_fabs(d); }
long double fabsl(long double d) { return __ESBMC_fabsl(d); }
float fabsf(float f) { return __ESBMC_fabsf(f); }
int isfinite(double d) { return __ESBMC_isfinite(d); }
int isinf(double d) { return __ESBMC_isinf(d); }
int isnan(double d) { return __ESBMC_isnan(d); }
int isnormal(double d) { return __ESBMC_isnormal(d); }
int signbit(double d) { return __ESBMC_sign(d); }

int __fpclassifyd(double d) {
  if(__ESBMC_isnan(d)) return FP_NAN;
  if(__ESBMC_isinf(d)) return FP_INFINITE;
  if(d==0) return FP_ZERO;
  if(__ESBMC_isnormal(d)) return FP_NORMAL;
  return FP_SUBNORMAL;
}

int __fpclassifyf(float f) {
  if(__ESBMC_isnan(f)) return FP_NAN;
  if(__ESBMC_isinf(f)) return FP_INFINITE;
  if(f==0) return FP_ZERO;
  if(__ESBMC_isnormal(f)) return FP_NORMAL;
  return FP_SUBNORMAL;
}

int __fpclassify(long double d) {
  if(__ESBMC_isnan(d)) return FP_NAN;
  if(__ESBMC_isinf(d)) return FP_INFINITE;
  if(d==0) return FP_ZERO;
  if(__ESBMC_isnormal(d)) return FP_NORMAL;
  return FP_SUBNORMAL;
}

int fegetround() { return __ESBMC_rounding_mode; }

int fesetround(int __rounding_direction) {
  __ESBMC_rounding_mode=__rounding_direction;
}

double inline fabs(double x) {
  return (x < 0) ? -x : x;
}

double fmod(double a, double b) {
  return a - (b * (int)(a/b));
}

double cos(double x)
{
    double t , s;
    int p;
    p = 0;
    s = 1.0;
    t = 1.0;
    x = fmod(x + M_PI, M_PI * 2) - M_PI; // restrict x so that -M_PI < x < M_PI
    double xsqr = x*x;
    double ab = 1;
    while((ab > PREC) && (p < 15))
    {
        p++;
        t = (-t * xsqr) / (((p<<1) - 1) * (p<<1));
        s += t;
        ab = (s==0) ? 1 : fabs(t/s);
    }
    return s;
}

double sin(double x)
{
   return cos(x-M_PI_2);
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
  //float e = 0.000001; /* e decides the accuracy level*/
  //double e = 1e-16;
  double e = 1;
  int i = 0;
//  while(fabs(x - y) > e)
  while(i++ < 15) //Change this line to increase precision
  {
    x = (x + y)/2.0;
    y = n/x;
  }
  return x;
}
