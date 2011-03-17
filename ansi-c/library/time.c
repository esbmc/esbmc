/* FUNCTION: time */

#ifndef __CPROVER_TIME_H_INCLUDED
#include <time.h>
#define __CPROVER_TIME_H_INCLUDED
#endif

#undef time

time_t time(time_t *tloc)
{
  time_t res;
  if(!tloc) *tloc=res;
  return res;
}

