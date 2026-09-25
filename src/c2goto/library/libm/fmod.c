#include <math.h>

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

fmod_def(float, fmodf, remainderf, fabsf);
fmod_def(double, fmod, remainder, fabs);
fmod_def(long double, fmodl, remainderl, fabsl);

#undef fmod_def
