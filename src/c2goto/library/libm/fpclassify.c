
#include <math.h>
#ifdef _MSVC
#  undef isnan
#  undef isinf

#  define classify_return_type short

#  define _signbit(type, name)                                                 \
    int name(type d)                                                           \
    {                                                                          \
    __ESBMC_HIDE:;                                                             \
      return __builtin_signbit((float)d);                                      \
    }

_signbit(double, _dsign);
_signbit(long double, _ldsign);
_signbit(float, _fdsign);
#  undef _signbit

/* MSVC's <math.h> builds isgreater() and the rest of the ordering macros on
 * these; unordered operands must yield 0, which the IEEE comparisons below
 * give for free since all three are false on NaN. */
#  define _pcomp(type, name)                                                   \
    int name(type x, type y)                                                   \
    {                                                                          \
    __ESBMC_HIDE:;                                                             \
      return x < y ? _FP_LT : x > y ? _FP_GT : x == y ? _FP_EQ : 0;            \
    }

_pcomp(double, _dpcomp);
_pcomp(long double, _ldpcomp);
_pcomp(float, _fdpcomp);
#  undef _pcomp
#else
#  define classify_return_type int
#endif

#define classify_def(type, name, isnan_func, isinf_func, isnormal_func)        \
  classify_return_type name(type f)                                            \
  {                                                                            \
  __ESBMC_HIDE:;                                                               \
    return isnan_func(f)      ? FP_NAN                                         \
           : isinf_func(f)    ? FP_INFINITE                                    \
           : f == 0           ? FP_ZERO                                        \
           : isnormal_func(f) ? FP_NORMAL                                      \
                              : FP_SUBNORMAL;                                  \
  }

classify_def(float, _fdclass, isnan, isinf, __builtin_isnormal);
classify_def(float, __fpclassifyf, isnan, isinf, __builtin_isnormal);

classify_def(double, _dclass, isnan, isinf, __builtin_isnormal);
classify_def(double, __fpclassify, isnan, isinf, __builtin_isnormal);
classify_def(double, __fpclassifyd, isnan, isinf, __builtin_isnormal);

classify_def(long double, _ldclass, isnan, isinf, __builtin_isnormal);
classify_def(long double, __fpclassifyl, isnan, isinf, __builtin_isnormal);

#undef classify_def
#undef classify_return_type
