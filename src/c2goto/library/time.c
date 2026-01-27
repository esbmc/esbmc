#ifdef _MINGW
#  define __declspec /* hacks */
#endif

#ifdef _MSVC
#  define _INC_TIME_INL
#  define time crt_time
#endif
#include <time.h>
#undef time
#undef difftime
#undef clock
#undef mktime
#undef localtime
#undef gmtime
#undef localtime_r
#undef gmtime_r
#undef asctime
#undef ctime
#undef strftime
#undef timespec_get

#ifndef TIME_UTC
#  define TIME_UTC 1
#endif

/* Nondet declarations for custom types (built-in nondet_int/long/etc.
   are auto-declared by the compiler, but time_t/clock_t need explicit decls) */
time_t __VERIFIER_nondet_time_t();
clock_t __VERIFIER_nondet_clock_t();

/* ── Ghost state for localtime/mktime roundtrip ── */
static struct tm __esbmc_localtime_buf;
static time_t __esbmc_localtime_last_input;

/* ── Ghost state for gmtime ── */
static struct tm __esbmc_gmtime_buf;

/* ── Static string buffers for asctime/ctime ── */
static char __esbmc_asctime_buf[26];
static char __esbmc_ctime_buf[26];

/* ── Helper: fill struct tm with constrained nondet values ── */
static void __esbmc_fill_tm_nondet(struct tm *t)
{
__ESBMC_HIDE:;
  t->tm_sec = nondet_int();
  __ESBMC_assume(t->tm_sec >= 0 && t->tm_sec <= 60);
  t->tm_min = nondet_int();
  __ESBMC_assume(t->tm_min >= 0 && t->tm_min <= 59);
  t->tm_hour = nondet_int();
  __ESBMC_assume(t->tm_hour >= 0 && t->tm_hour <= 23);
  t->tm_mday = nondet_int();
  __ESBMC_assume(t->tm_mday >= 1 && t->tm_mday <= 31);
  t->tm_mon = nondet_int();
  __ESBMC_assume(t->tm_mon >= 0 && t->tm_mon <= 11);
  t->tm_year = nondet_int();
  t->tm_wday = nondet_int();
  __ESBMC_assume(t->tm_wday >= 0 && t->tm_wday <= 6);
  t->tm_yday = nondet_int();
  __ESBMC_assume(t->tm_yday >= 0 && t->tm_yday <= 365);
  t->tm_isdst = nondet_int();
  __ESBMC_assume(t->tm_isdst >= -1 && t->tm_isdst <= 1);
}

/* ── time ── */
time_t time(time_t *tloc)
{
__ESBMC_HIDE:;
  time_t res = __VERIFIER_nondet_time_t();
  if (tloc)
    *tloc = res;
  return res;
}

/* ── difftime ── */
double difftime(time_t end, time_t beginning)
{
__ESBMC_HIDE:;
  if (end >= beginning)
    return (double)(end - beginning);
  else
    return -(double)(beginning - end);
}

/* ── clock ── */
clock_t clock(void)
{
__ESBMC_HIDE:;
  return __VERIFIER_nondet_clock_t();
}

/* ── localtime ── */
struct tm *localtime(const time_t *timer)
{
__ESBMC_HIDE:;
  __ESBMC_assert(timer != NULL, "localtime argument must not be NULL");
  __esbmc_localtime_last_input = *timer;
  __esbmc_fill_tm_nondet(&__esbmc_localtime_buf);
  return &__esbmc_localtime_buf;
}

/* ── gmtime ── */
struct tm *gmtime(const time_t *timer)
{
__ESBMC_HIDE:;
  __ESBMC_assert(timer != NULL, "gmtime argument must not be NULL");
  __esbmc_fill_tm_nondet(&__esbmc_gmtime_buf);
  return &__esbmc_gmtime_buf;
}

/* ── localtime_r ── */
struct tm *localtime_r(const time_t *timer, struct tm *buf)
{
__ESBMC_HIDE:;
  __ESBMC_assert(timer != NULL, "localtime_r timer must not be NULL");
  __ESBMC_assert(buf != NULL, "localtime_r result buffer must not be NULL");
  __esbmc_fill_tm_nondet(buf);
  return buf;
}

/* ── gmtime_r ── */
struct tm *gmtime_r(const time_t *timer, struct tm *buf)
{
__ESBMC_HIDE:;
  __ESBMC_assert(timer != NULL, "gmtime_r timer must not be NULL");
  __ESBMC_assert(buf != NULL, "gmtime_r result buffer must not be NULL");
  __esbmc_fill_tm_nondet(buf);
  return buf;
}

/* ── mktime ── */
time_t mktime(struct tm *timeptr)
{
__ESBMC_HIDE:;
  __ESBMC_assert(timeptr != NULL, "mktime argument must not be NULL");
  if (timeptr == NULL)
    return (time_t)-1;

  /* Normalize tm_mon into [0,11], adjusting tm_year (closed-form) */
  int m = timeptr->tm_mon;
  if (m < 0 || m > 11)
  {
    int years;
    if (m >= 0)
      years = m / 12;
    else
      years = -((-m + 11) / 12);
    timeptr->tm_year += years;
    timeptr->tm_mon = m - years * 12;
  }

  /* Roundtrip consistency: if called on the localtime static buffer,
     return the original time_t that produced it */
  if (timeptr == &__esbmc_localtime_buf)
    return __esbmc_localtime_last_input;

  return __VERIFIER_nondet_time_t();
}

/* ── asctime ── */
char *asctime(const struct tm *timeptr)
{
__ESBMC_HIDE:;
  __ESBMC_assert(timeptr != NULL, "asctime argument must not be NULL");
  __esbmc_asctime_buf[25] = '\0';
  return __esbmc_asctime_buf;
}

/* ── ctime ── */
char *ctime(const time_t *timer)
{
__ESBMC_HIDE:;
  __ESBMC_assert(timer != NULL, "ctime argument must not be NULL");
  __esbmc_ctime_buf[25] = '\0';
  return __esbmc_ctime_buf;
}

/* ── strftime ── */
size_t
strftime(char *s, size_t maxsize, const char *format, const struct tm *timeptr)
{
__ESBMC_HIDE:;
  __ESBMC_assert(format != NULL, "strftime format must not be NULL");
  __ESBMC_assert(timeptr != NULL, "strftime timeptr must not be NULL");

  if (maxsize == 0)
    return 0;

  size_t result = nondet_uint();
  __ESBMC_assume(result <= maxsize - 1);
  if (result > 0)
    s[result] = '\0';
  return result;
}

/* ── timespec_get (C11) ── */
int timespec_get(struct timespec *ts, int base)
{
__ESBMC_HIDE:;
  __ESBMC_assert(ts != NULL, "timespec_get ts must not be NULL");
  if (base == TIME_UTC)
  {
    ts->tv_sec = nondet_long();
    ts->tv_nsec = nondet_long();
    __ESBMC_assume(ts->tv_nsec >= 0 && ts->tv_nsec <= 999999999L);
    return TIME_UTC;
  }
  return 0;
}
