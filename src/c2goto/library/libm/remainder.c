#define __CRT__NO_INLINE /* Don't let mingw insert code */

#include <math.h>

#define remainder_def(type, name, remquo_func)                                 \
  type name(type x, type y)                                                    \
  {                                                                            \
  __ESBMC_HIDE:;                                                               \
    int quo;                                                                   \
    return remquo_func(x, y, &quo);                                            \
  }                                                                            \
                                                                               \
  type __##name(type x, type y)                                                \
  {                                                                            \
  __ESBMC_HIDE:;                                                               \
    return name(x, y);                                                         \
  }

remainder_def(float, remainderf, remquof);
remainder_def(double, remainder, remquo);
remainder_def(long double, remainderl, remquol);

#undef remainder_def
