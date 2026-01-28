
#pragma once

#include <__esbmc/stddefs.h>
#include <stddef.h> /* size_t */

__ESBMC_C_CPP_BEGIN

#ifndef __time_t_defined
#define __time_t_defined 1
#define _TIME_T /* macOS guard */
/* On Windows, time_t is already defined as __time64_t (long long) by
   corecrt.h, which is pulled in transitively via #include_next <stddef.h>.
   Redefining it as long would conflict since long is 32-bit on Windows. */
#ifndef _WIN32
typedef long time_t;
#endif
#endif

#ifndef __clock_t_defined
#define __clock_t_defined 1
#define _CLOCK_T /* macOS guard */
typedef long clock_t;
#endif

#define CLOCKS_PER_SEC 1000000L
#define TIME_UTC 1

#ifndef NULL
#  ifdef __cplusplus
#    define NULL 0
#  else
#    define NULL ((void *)0)
#  endif
#endif

#ifndef _STRUCT_TIMESPEC
#define _STRUCT_TIMESPEC 1
struct timespec
{
  time_t tv_sec;
  long tv_nsec;
};
#endif

#ifndef __struct_tm_defined
#define __struct_tm_defined 1
struct tm
{
  int tm_sec;
  int tm_min;
  int tm_hour;
  int tm_mday;
  int tm_mon;
  int tm_year;
  int tm_wday;
  int tm_yday;
  int tm_isdst;
};
#endif

time_t time(time_t *tloc);
double difftime(time_t end, time_t beginning);
clock_t clock(void);
time_t mktime(struct tm *timeptr);

struct tm *localtime(const time_t *timer);
struct tm *gmtime(const time_t *timer);
struct tm *localtime_r(const time_t *timer, struct tm *buf);
struct tm *gmtime_r(const time_t *timer, struct tm *buf);

char *asctime(const struct tm *timeptr);
char *ctime(const time_t *timer);

size_t strftime(char *s, size_t maxsize, const char *format,
                const struct tm *timeptr);

int timespec_get(struct timespec *ts, int base);

__ESBMC_C_CPP_END
