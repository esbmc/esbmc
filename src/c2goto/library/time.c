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

time_t time(time_t *tloc)
{
__ESBMC_HIDE:;
  time_t res = __VERIFIER_nondet_time_t();
  if (tloc)
    *tloc = res;
  return res;
}
