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
#define M_PI_2   1.57079632679489661923      // Pi/2
#define PREC 1e-16
#define M_LN10   2.30258509299404568402
#define DBL_EPSILON 2.2204460492503131e-16

#define M_E     2.7182818284590452354
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

int __fpclassify(double d) {
  if(__ESBMC_isnan(d)) return FP_NAN;
  if(__ESBMC_isinf(d)) return FP_INFINITE;
  if(d==0) return FP_ZERO;
  if(__ESBMC_isnormal(d)) return FP_NORMAL;
  return FP_SUBNORMAL;
}

int fegetround() { return __ESBMC_rounding_mode; }

int fesetround(int __rounding_direction) {
  __ESBMC_rounding_mode=__rounding_direction;
  return __ESBMC_rounding_mode;
}

double fabs(double x) {
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

//#include "math.h"
//#include <assert.h>
/*
 * absolute constants
 */

#define PI 3.14159265358979324          /* pi                           */
#define HALF_PI 1.57079632679489662     /* pi/2                         */
#define REC_PI 0.318309886183790672     /* 1/pi                         */
#define RPIBY2 0.636619772367581343     /* reciprocal of pi/2           */
#define E 2.718281828459045235          /* e                            */
#define LOGBE2 0.69314718055994530942   /* log of 2 to base e           */
#define LOGB2E 1.44269504088896341      /* log of e to base 2           */
#define ROOT_2 1.4142135623730950488    /* square root of 2             */
#define ROOT_3  1.732050807568877293    /* square root of 3             */

/*
 * checked octal constants
 */

#define PI_BY_2 1.57079632679489662     /* pi/2                         */
#define PI_BY_4 0.785398163397448310    /* pi/4                         */
#define LOGB10E 0.434294481903251828    /* log of e to base 10          */
#define ROOT_05 0.70710678118654752440  /* square root of 0.5           */

/*
 * system dependent constants
 */

#define TINY 0.293873587705571877E-38   /* smallest no = 2**-128        */
#define HUGE 3.40282347e+38F   /* largest no = 2**+127         */
#define LOG_HUGE 0.880296919311130543E+02       /* log of HUGE          */
#define LOG_TINY -0.887228391116729997E+02      /* log of TINY          */
#define MIN_EXP -128                    /* minimum base 2 exponent      */
#define MAX_EXP 127                     /* maximum base 2 exponent      */
#define MAXLONG 017777777777L           /* largest long integer         */
#define SIGFIGS 18                      /* max no useful digits in dtoa */
#define TRIG_MAX 3.1415926535897932385e12/* arg limit for trig functions*/
#define ROOT_EPS 0.372529029846191406e-8        /* 2**-(t/2), t = 56    */
#define REC_HUGE 0.587747175411143754E-38/* 2**-127 = 1 / HUGE          */

/*
 * error codes to communicate with cmathe
 */

#define FP_OPER 1               /* FPU op code error                    */
#define FP_ZDIV 2               /* FPU divide by zero                   */
#define FP_FTOI 3               /* FPU float to integer conv error      */
#define FP_OFLO 4               /* FPU overflow                         */
#define FP_UFLO 5               /* FPU underflow                        */
#define FP_UDEF 6               /* FPU undefined variable (-0)          */
#define FP_BIGI 7               /* Atof input too large                 */
#define FP_BADC 8               /* Bad character in atof input string   */
#define FP_NESQ 9               /* Square root of negative number       */
#define FP_LEXP 10              /* Exp argument too large               */
#define FP_SEXP 11              /* Exp argument too small               */
#define FP_NLOG 12              /* Log argument zero or negative        */
#define FP_TANE 13              /* Argument of tan too large            */
#define FP_TRIG 14              /* Argument of sin/cos too large        */
#define FP_ATAN 15              /* Atan2 arguments both zero            */
#define FP_COTE 16              /* Argument of cotan too small          */
#define FP_ARSC 17              /* Bad argument for asin/acos           */
#define FP_SINH 18              /* Argument of sinh too large           */
#define FP_COSH 19              /* Argument of cosh too large           */
#define FP_POWN 20              /* Negative argument in pow             */
#define FP_POWO 21              /* Result of pow overflows              */
#define FP_POWU 22              /* Result of pow underflows             */

/* numbers of each type of error - used to determine argument type      */

#define FP_NFPU 6               /* No of FPU generated errors           */
#define FP_NSTR 2               /* No of string argument errors         */
#define FP_NMAR 14              /* No of math routine double arg errors */

/* The following define error codes which are assigned to $$ferr by
 * the cmathe error package. They will be flagged as unknown errors by
 * the perror() function and these error numbers printed.  Normally that
 * error reporting mechanism will not be used, it has been included for
 * benefit of programs which have been transported from systems which
 * use the perror() function.                                           */

#define SIGFPE 108      /* floating point exception error               */
#define EDOM 133        /* domain error (input argument inadmissable)   */
#define ERANGE 134      /* range error (result too large or small)      */

static double  p0 = -0.136887688941919269e2;
static double  p1 = -0.205058551958616520e2;
static double  p2 = -0.849462403513206835e1;
static double  p3 = -0.837582993681500593e0;
static double  q0 =  0.410663066825757813e2;
static double  q1 =  0.861573495971302425e2;
static double  q2 =  0.595784361425973445e2;
static double  q3 =  0.150240011600285761e2;
static double a[] = {
    0.0,
    0.523598775598298873,               /* pi / 6                       */
    PI_BY_2,                            /* pi / 2                       */
    1.047197551196597746                /* pi / 3                       */
};

#define CON1 0.267949192431122706       /* 2 - sqrt(3)                  */
#define ROOT_3M1 0.732050807568877294   /* sqrt(3) - 1                  */
static double _atan(double f, int n)
{
    double g, q, r;

    if (f > CON1) {
        f = (((ROOT_3M1 * f - 0.5) - 0.5) + f) / (ROOT_3 + f);
        n++;
    }
    if (f > ROOT_EPS || f < -ROOT_EPS) {
        g = f * f;
        q = (((g + q3)*g + q2)*g + q1)*g + q0;
        r = (((p3*g + p2)*g + p1)*g + p0)*g / q;
        f = f + f * r;
    }
    if (n > 1)
        f = -f;
    return(f + a[n]);
}


double atan(double x)
{
    double a;

    a = x < 0.0 ? -x : x;
    if (a > 1.0)
        a = _atan(1.0 / a, 2);
    else
        a = _atan(a, 0);
    return(x < 0.0 ? -a : a);
}


double atan2(double v, double u)
{
    //double _atan(), au, av, f;
    double au, av, f;

    av = v < 0.0 ? -v : v;
    au = u < 0.0 ? -u : u;
    if (u != 0.0) {
        if (av > au) {
            if ((f = au / av) == 0.0)
                f = HALF_PI;
            else
                f = _atan(f, 2);
        }
        else {
            if ((f = av / au) == 0.0)
                f = 0.0;
            else
                f = _atan(f, 0);
        }
    }
    else {
        if (v != 0)
            f = HALF_PI;
        else {
            //cmemsg(FP_ATAN, &v);
            //assert(0);
            f = 0.0;
        }
    }
    if (u < 0.0)
        f = PI - f;
    return(v < 0.0 ? -f : f);
}

double pow(double base, double exponent){
   int result = 1;
   if (exponent == 0)
      return result;
   if (exponent < 0)
      return 1 / pow(base, -exponent);
   float temp = pow(base, exponent / 2);
   if ((int)exponent % 2 == 0)
      return temp * temp;
   else
      return (base * temp * temp);
}
