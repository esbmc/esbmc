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
#define PREC 1e-15
#define M_LN10   2.30258509299404568402
#define DBL_EPSILON 2.2204460492503131e-16

#define M_E     2.71828182845905
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
    double t , s ;
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
  while(i++ < 10) //Change this line to increase precision
  {
    x = (x + y)/2.0;
    y = n/x;
  }
  return x;
}

static double _expi_square_tbl[11] = {
        M_E,            // e^1
        M_E2,           // e^2
        M_E4,           // e^4
        M_E8,           // e^8
        M_E16,          // e^16
        M_E32,          // e^32
        M_E64,          // e^64
        M_E128,         // e^128
        M_E256,         // e^256
        M_E512,         // e^512
        M_E1024,        // e^1024
};

static double _expi(int n) {
        int i;
        double val;

        if (n > 1024) {
                //return FP_INFINITE;
            return (1.0/0.0);
        }

        val = 1.0;

        for (i = 0; n; i++) {
                if (n & (1 << i)) {
                        n &= ~(1 << i);
                        val *= _expi_square_tbl[i];
                }
        }

        return val;
}

static double _dbl_inv_fact[] = {
        1.0 / 1.0,                                      // 1 / 0!
        1.0 / 1.0,                                      // 1 / 1!
        1.0 / 2.0,                                      // 1 / 2!
        1.0 / 6.0,                                      // 1 / 3!
        1.0 / 24.0,                                     // 1 / 4!
        1.0 / 120.0,                            // 1 / 5!
        1.0 / 720.0,                            // 1 / 6!
        1.0 / 5040.0,                           // 1 / 7!
        1.0 / 40320.0,                          // 1 / 8!
        1.0 / 362880.0,                         // 1 / 9!
        1.0 / 3628800.0,                        // 1 / 10!
        1.0 / 39916800.0,                       // 1 / 11!
        1.0 / 479001600.0,                      // 1 / 12!
        1.0 / 6227020800.0,                     // 1 / 13!
        1.0 / 87178291200.0,            // 1 / 14!
        1.0 / 1307674368000.0,          // 1 / 15!
        1.0 / 20922789888000.0,         // 1 / 16!
        1.0 / 355687428096000.0,        // 1 / 17!
        1.0 / 6402373705728000.0,       // 1 / 18!
};

double exp(double x) {
        int int_part;
        int invert;
        double value;
        double x0;
        int i;

        if (x == 0) {
                return 1;
        }
        else if (x < 0) {
                invert = 1;
                x = -x;
        }
        else {
                invert = 0;
        }

        /* extract integer component */
        int_part = (int) x;

        /* set x to fractional component */
        x -= (double) int_part;

        /* perform Taylor series approximation with nineteen terms */
        value = 0.0;
        x0 = 1.0;
        for (i = 0; i < 19; i++) {
                value += x0 * _dbl_inv_fact[i];
                x0 *= x;
        }

        /* multiply by exp of the integer component */
        value *= _expi(int_part);

        if (invert) {
                return (1.0 / value);
        }
        else {
                return value;
        }
}

double log(double x) {
        double y, y_old, ey, epsilon;

        y = 0.0;
        y_old = 1.0;
        epsilon = DBL_EPSILON;
        int i = 0;
        while ((y > y_old + epsilon || y < y_old - epsilon) && (i++ < 20)) {
                y_old = y;
                ey = exp(y);
                y -= (ey - x) / ey;

                if (y > 700.0) {
                        y = 700.0;
                }
                if (y < -700.0) {
                        y = -700.0;
                }

                epsilon = (fabs(y) > 1.0) ? fabs(y) * DBL_EPSILON : DBL_EPSILON;
        }

        if (y == 700.0) {
                return FP_INFINITE;
            return (1.0/0.0);
        }
        if (y == -700.0) {
                return FP_INFINITE;
            return (1.0/0.0);
        }

        return y;
}

double log10(double x) {
        return (log(x) / M_LN10);
}
