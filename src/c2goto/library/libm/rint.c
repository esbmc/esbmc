#include <fenv.h>
#include <math.h>

/* Under fixedbv, nearbyint is not lowered to the floating-point node, so it
 * needs a body. The cast truncates toward zero and the fraction is exact. */
#define fixedbv_nearbyint_def(type, name)                                      \
  type name(type x)                                                            \
  {                                                                            \
  __ESBMC_HIDE:;                                                               \
    long long t = (long long)x;                                                \
    type frac = x - t;                                                         \
    switch (fegetround())                                                      \
    {                                                                          \
    case FE_TOWARDZERO:                                                        \
      return t;                                                                \
    case FE_DOWNWARD:                                                          \
      return frac < 0 ? t - 1 : t;                                             \
    case FE_UPWARD:                                                            \
      return frac > 0 ? t + 1 : t;                                             \
    default:                                                                   \
      if (frac > 0.5 || (frac == 0.5 && (t & 1)))                              \
        return t + 1;                                                          \
      if (frac < -0.5 || (frac == -0.5 && (t & 1)))                            \
        return t - 1;                                                          \
      return t;                                                                \
    }                                                                          \
  }                                                                            \
                                                                               \
  type __##name(type x)                                                        \
  {                                                                            \
  __ESBMC_HIDE:;                                                               \
    return name(x);                                                            \
  }

#ifdef __ESBMC_FIXEDBV
fixedbv_nearbyint_def(float, nearbyintf);
fixedbv_nearbyint_def(double, nearbyint);
fixedbv_nearbyint_def(long double, nearbyintl);
#endif

#undef fixedbv_nearbyint_def

#define rint_def(ret_type, type, name, isnan_func)                             \
  ret_type name(type f)                                                        \
  {                                                                            \
  __ESBMC_HIDE:;                                                               \
    return isnan_func(f);                                                      \
  }                                                                            \
                                                                               \
  ret_type __##name(type f)                                                    \
  {                                                                            \
  __ESBMC_HIDE:;                                                               \
    return name(f);                                                            \
  }

rint_def(float, float, rintf, nearbyintf);
rint_def(double, double, rint, nearbyint);
rint_def(long double, long double, rintl, nearbyintl);

rint_def(long, float, lrintf, nearbyintf);
rint_def(long, double, lrint, nearbyint);
rint_def(long, long double, lrintl, nearbyintl);

rint_def(long long, float, llrintf, nearbyintf);
rint_def(long long, double, llrint, nearbyint);
rint_def(long long, long double, llrintl, nearbyintl);

#undef rint_def
