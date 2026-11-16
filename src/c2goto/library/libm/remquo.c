#include <fenv.h>
#include <math.h>

/* Under fixedbv, x and y are integers scaled by the same power of two, so
 * the quotient and remainder of the scaled integers are exact; round the
 * quotient to nearest, ties to even (C17 7.12.10.2), on their magnitudes. */
#define fixedbv_remquo_def(type, name, itype, utype)                           \
  type name(type x, type y, int *quo)                                          \
  {                                                                            \
  __ESBMC_HIDE:;                                                               \
    if (y == 0.0)                                                              \
      return NAN;                                                              \
                                                                               \
    union                                                                      \
    {                                                                          \
      type f;                                                                  \
      itype i;                                                                 \
    } a = {x}, b = {y}, r;                                                     \
    utype am = a.i < 0 ? -(utype)a.i : (utype)a.i;                             \
    utype bm = b.i < 0 ? -(utype)b.i : (utype)b.i;                             \
    utype qm = am / bm;                                                        \
    utype rm = am % bm;                                                        \
    int negative = a.i < 0;                                                    \
    if (rm > bm - rm || (rm == bm - rm && (qm & 1)))                           \
    {                                                                          \
      qm += 1;                                                                 \
      rm = bm - rm;                                                            \
      negative = !negative;                                                    \
    }                                                                          \
    r.i = negative ? -(itype)rm : (itype)rm;                                   \
                                                                               \
    /* C17 7.12.10.3: the quotient's sign and its low bits. */                 \
    int q = (int)(qm & 0x7fffffff);                                            \
    *quo = (a.i < 0) != (b.i < 0) ? -q : q;                                    \
    return r.f;                                                                \
  }                                                                            \
                                                                               \
  type __##name(type x, type y, int *quo)                                      \
  {                                                                            \
  __ESBMC_HIDE:;                                                               \
    return name(x, y, quo);                                                    \
  }

/* remquo: the remainder is exactly what remainder() returns -- normally the
 * solver's fp.rem via frontend interception, with remainder.c's
 * self-contained body as the fallback (that body never calls back here, so
 * the chain cannot recurse).
 *
 * *quo receives the sign and low bits of the rounded-to-nearest integral
 * quotient x/y (C17 7.12.10.3 requires at least three bits). It is computed
 * with llrint(x/y), which double-rounds and can overflow long long for huge
 * quotient magnitudes; C only defines *quo modulo a power of two, so the
 * low bits remain meaningful in the common range. */
#define remquo_def(                                                            \
  type, name, isnan_func, isinf_func, llrint_func, remainder_func)             \
  type name(type x, type y, int *quo)                                          \
  {                                                                            \
  __ESBMC_HIDE:;                                                               \
    if (isnan_func(x) || isnan_func(y))                                        \
      return NAN;                                                              \
                                                                               \
    if (y == 0.0)                                                              \
      return NAN;                                                              \
                                                                               \
    if (isinf_func(x))                                                         \
      return NAN;                                                              \
                                                                               \
    if (isinf_func(y))                                                         \
    {                                                                          \
      *quo = 0;                                                                \
      return x;                                                                \
    }                                                                          \
                                                                               \
    int old_rm = fegetround();                                                 \
    fesetround(FE_TONEAREST);                                                  \
    long long rquot = llrint_func(x / y);                                      \
    fesetround(old_rm);                                                        \
                                                                               \
    *quo = (int)rquot;                                                         \
    return remainder_func(x, y);                                               \
  }                                                                            \
                                                                               \
  type __##name(type x, type y, int *quo)                                      \
  {                                                                            \
  __ESBMC_HIDE:;                                                               \
    return name(x, y, quo);                                                    \
  }

#ifdef __ESBMC_FIXEDBV
fixedbv_remquo_def(float, remquof, int, unsigned);
fixedbv_remquo_def(double, remquo, long long, unsigned long long);
#  if __SIZEOF_POINTER__ == 8 /* 32-bit fixedbv long double is not 64 bits */
fixedbv_remquo_def(long double, remquol, long long, unsigned long long);
#  else
remquo_def(long double, remquol, isnan, isinf, llrintl, remainderl);
#  endif
#else
remquo_def(float, remquof, isnan, isinf, llrintf, remainderf);
remquo_def(double, remquo, isnan, isinf, llrint, remainder);
remquo_def(long double, remquol, isnan, isinf, llrintl, remainderl);
#endif

#undef fixedbv_remquo_def
#undef remquo_def
