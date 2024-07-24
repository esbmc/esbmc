#ifdef _MINGW

#  define __declspec /* hacks */
#endif

#ifdef _MSVC
#  define _INC_TIME_INL
#  define time crt_time
#endif
#include <time.h>
#undef time

time_t __VERIFIER_nondet_time_t();

/* mingw has a static inline definition of time() in <time.h> delegating to
 * either _time32 or _time64 */
#if defined(__MINGW64__)
time_t _time64(time_t *tloc)
#elif defined(__MINGW32__)
time_t _time32(time_t *tloc)
#else
time_t time(time_t *tloc)
#endif
{
__ESBMC_HIDE:;
  time_t res = __VERIFIER_nondet_time_t();
  if (tloc)
    *tloc = res;
  return res;
}
