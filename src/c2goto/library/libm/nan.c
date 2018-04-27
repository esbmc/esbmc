#define __CRT__NO_INLINE /* Don't let mingw insert code */

#include <math.h>

#ifdef __APPLE__
#define __nan_def(type, name) ;

#else
#define __nan_def(type, name)                                                  \
  type __##name(const char *arg)                                               \
  {                                                                            \
  __ESBMC_HIDE:;                                                               \
    return name(arg);                                                          \
  }
#endif

#define nan_def(type, name)                                                    \
  type name(const char *arg)                                                   \
  {                                                                            \
  __ESBMC_HIDE:                                                                \
    (void)arg;                                                                 \
    return NAN;                                                                \
  }                                                                            \
  __nan_def(type, name)

nan_def(float, nanf);
nan_def(double, nan);
nan_def(long double, nanl);

#undef nan_def
#undef __nan_def
