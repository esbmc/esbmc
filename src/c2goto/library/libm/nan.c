#define __CRT__NO_INLINE /* Don't let mingw insert code */

#include <math.h>
#include "../intrinsics.h"

#define nan_def(type, name)                                                    \
  type name(const char *arg)                                                   \
  {                                                                            \
  __ESBMC_HIDE:                                                                \
    (void)arg;                                                                 \
    return NAN;                                                                \
  }                                                                            \
                                                                               \
  type __##name(const char *arg)                                               \
  {                                                                            \
  __ESBMC_HIDE:;                                                               \
    return name(arg);                                                          \
  }

nan_def(float, nanf);
// avoid conflicting types for '__nan' on the Mac
#ifndef __APPLE__
nan_def(double, nan);
#endif
nan_def(long double, nanl);

#undef nan_def
