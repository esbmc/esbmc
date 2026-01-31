
#pragma once

#include <__esbmc/stddefs.h>
#include <stddef.h> /* size_t */

__ESBMC_C_CPP_BEGIN

#ifndef __time_t_defined
#define __time_t_defined 1
#define _TIME_T /* macOS guard */
/* On Windows, time_t is __time64_t (long long) because long is 32-bit.
   On Unix-like systems, time_t is typically long (which is 64-bit on LP64). */
#ifdef _WIN32
typedef long long time_t;
#else
typedef long time_t;
#endif
#endif

#ifndef __clock_t_defined
#define __clock_t_defined 1
#define _CLOCK_T /* macOS guard */
typedef long clock_t;
#endif

#ifndef __clockid_t_defined
#define __clockid_t_defined 1
typedef int clockid_t;
#endif

#ifndef __timer_t_defined
#define __timer_t_defined 1
typedef void *timer_t;
#endif

#ifndef __pid_t_defined
#define __pid_t_defined 1
typedef int pid_t;
#endif

#define CLOCKS_PER_SEC 1000000L
#define TIME_UTC 1

/* POSIX clock IDs */
#define CLOCK_REALTIME 0
#define CLOCK_MONOTONIC 1
#define CLOCK_PROCESS_CPUTIME_ID 2
#define CLOCK_THREAD_CPUTIME_ID 3
#define CLOCK_MONOTONIC_RAW 4
#define CLOCK_REALTIME_COARSE 5
#define CLOCK_MONOTONIC_COARSE 6
#define CLOCK_BOOTTIME 7

/* Flags for clock_nanosleep and timer_settime */
#define TIMER_ABSTIME 1

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

#ifndef _STRUCT_ITIMERSPEC
#define _STRUCT_ITIMERSPEC 1
struct itimerspec
{
  struct timespec it_interval;
  struct timespec it_value;
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
  /* glibc extensions */
  long tm_gmtoff;
  const char *tm_zone;
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

/* POSIX reentrant versions */
char *asctime_r(const struct tm *timeptr, char *buf);
char *ctime_r(const time_t *timer, char *buf);

size_t strftime(char *s, size_t maxsize, const char *format,
                const struct tm *timeptr);

/* POSIX string parsing */
char *strptime(const char *s, const char *format, struct tm *tm);

int timespec_get(struct timespec *ts, int base);

/* POSIX clock functions */
int clock_getres(clockid_t clk_id, struct timespec *res);
int clock_gettime(clockid_t clk_id, struct timespec *tp);
int clock_settime(clockid_t clk_id, const struct timespec *tp);
int clock_nanosleep(clockid_t clk_id, int flags, const struct timespec *request,
                    struct timespec *remain);

/* POSIX sleep functions */
int nanosleep(const struct timespec *req, struct timespec *rem);

/* POSIX timers */
struct sigevent; /* Forward declaration from signal.h */
int timer_create(clockid_t clockid, struct sigevent *sevp, timer_t *timerid);
int timer_delete(timer_t timerid);
int timer_settime(timer_t timerid, int flags, const struct itimerspec *new_value,
                  struct itimerspec *old_value);
int timer_gettime(timer_t timerid, struct itimerspec *curr_value);
int timer_getoverrun(timer_t timerid);

/* POSIX clock CPU-time */
int clock_getcpuclockid(pid_t pid, clockid_t *clock_id);

/* POSIX timezone functions */
void tzset(void);

/* POSIX getdate (XSI) */
struct tm *getdate(const char *string);
__ESBMC_EXTERN_NOVAL extern int getdate_err;

/* POSIX timezone variables */
__ESBMC_EXTERN_NOVAL extern int daylight;
__ESBMC_EXTERN_NOVAL extern long timezone;
__ESBMC_EXTERN_NOVAL extern char *tzname[2];

__ESBMC_C_CPP_END
