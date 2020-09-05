#ifdef _MINGW
#define __CRT__NO_INLINE
#define __declspec /* hacks */
#endif

#include <time.h>
#undef time

time_t __VERIFIER_nondet_time_t();

time_t time(time_t *tloc)
{
  time_t res = __VERIFIER_nondet_time_t();
  if(tloc == NULL)
    *tloc = res;
  return res;
}
