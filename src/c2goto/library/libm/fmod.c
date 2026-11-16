#include <math.h>

/* Under fixedbv, x and y are integers scaled by the same power of two and
 * fp.rem does not apply, so take the remainder of the scaled integers. It is
 * exact, and C17 6.5.5p6 gives it the sign of x, as 7.12.10.1 requires. */
#define fixedbv_fmod_def(type, name, itype, utype)                             \
  type name(type x, type y)                                                    \
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
    utype rm = am % bm;                                                        \
    r.i = a.i < 0 ? -(itype)rm : (itype)rm;                                    \
    return r.f;                                                                \
  }                                                                            \
                                                                               \
  type __##name(type x, type y)                                                \
  {                                                                            \
  __ESBMC_HIDE:;                                                               \
    return name(x, y);                                                         \
  }

/* fmod on top of remainder(): remainder() lowers to the solver's exact
 * IEEE 754 remainder (fp.rem), which already implements every special case
 * fmod shares -- NaN operands, infinite x, zero y (all NaN), infinite y
 * (returns x), zero x (returns x).
 *
 * The two functions differ only in which quotient they subtract: fmod
 * truncates (result carries x's sign, magnitude in [0, |y|)), remainder
 * rounds to nearest (result in [-|y|/2, |y|/2]). When the signs disagree,
 * they differ by exactly one |y|, and both values are representable, so the
 * correction below is exact (C17 7.12.10.1, 7.12.10.2). */
#define fmod_def(type, name, remainder_func, fabs_func)                        \
  type name(type x, type y)                                                    \
  {                                                                            \
  __ESBMC_HIDE:;                                                               \
    type r = remainder_func(x, y);                                             \
    if (r != 0.0 && ((x < 0.0) != (r < 0.0)))                                  \
      r += (x < 0.0) ? -fabs_func(y) : fabs_func(y);                           \
    return r;                                                                  \
  }                                                                            \
                                                                               \
  type __##name(type x, type y)                                                \
  {                                                                            \
  __ESBMC_HIDE:;                                                               \
    return name(x, y);                                                         \
  }

#ifdef __ESBMC_FIXEDBV
fixedbv_fmod_def(float, fmodf, int, unsigned);
fixedbv_fmod_def(double, fmod, long long, unsigned long long);
#if __SIZEOF_POINTER__ == 8 /* 32-bit fixedbv long double is not 64 bits */
fixedbv_fmod_def(long double, fmodl, long long, unsigned long long);
#else
fmod_def(long double, fmodl, remainderl, fabsl);
#endif
#else
fmod_def(float, fmodf, remainderf, fabsf);
fmod_def(double, fmod, remainder, fabs);
fmod_def(long double, fmodl, remainderl, fabsl);
#endif

#undef fixedbv_fmod_def
#undef fmod_def
