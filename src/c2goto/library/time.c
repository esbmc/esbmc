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
#undef asctime_r
#undef ctime_r
#undef strftime
#undef strptime
#undef timespec_get
#undef clock_getres
#undef clock_gettime
#undef clock_settime
#undef clock_nanosleep
#undef nanosleep
#undef timer_create
#undef timer_delete
#undef timer_settime
#undef timer_gettime
#undef timer_getoverrun
#undef clock_getcpuclockid
#undef tzset
#undef getdate

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

/* ── asctime_r (POSIX reentrant) ── */
char *asctime_r(const struct tm *timeptr, char *buf)
{
__ESBMC_HIDE:;
  __ESBMC_assert(timeptr != NULL, "asctime_r timeptr must not be NULL");
  __ESBMC_assert(buf != NULL, "asctime_r buf must not be NULL");
  buf[25] = '\0';
  return buf;
}

/* ── ctime_r (POSIX reentrant) ── */
char *ctime_r(const time_t *timer, char *buf)
{
__ESBMC_HIDE:;
  __ESBMC_assert(timer != NULL, "ctime_r timer must not be NULL");
  __ESBMC_assert(buf != NULL, "ctime_r buf must not be NULL");
  buf[25] = '\0';
  return buf;
}

/* ── strptime (POSIX string parsing) ── */
char *strptime(const char *s, const char *format, struct tm *tm)
{
__ESBMC_HIDE:;
  __ESBMC_assert(s != NULL, "strptime s must not be NULL");
  __ESBMC_assert(format != NULL, "strptime format must not be NULL");
  __ESBMC_assert(tm != NULL, "strptime tm must not be NULL");

  /* Nondeterministically fill tm with valid values */
  __esbmc_fill_tm_nondet(tm);

  /* Return either NULL (parse failure) or pointer into s (success) */
  _Bool success = nondet_bool();
  if (success)
  {
    /* On success, return pointer to first unprocessed character.
       For soundness, we return s (conservative: might have parsed nothing) */
    return (char *)s;
  }
  return (char *)0;
}

/* ── Helper: fill timespec with constrained nondet values ── */
static void __esbmc_fill_timespec_nondet(struct timespec *ts)
{
__ESBMC_HIDE:;
  ts->tv_sec = nondet_long();
  ts->tv_nsec = nondet_long();
  __ESBMC_assume(ts->tv_nsec >= 0 && ts->tv_nsec <= 999999999L);
}

/* ── clock_getres (POSIX) ── */
int clock_getres(clockid_t clk_id, struct timespec *res)
{
__ESBMC_HIDE:;
  /* clk_id validation: only known clock IDs are valid */
  _Bool valid_clock =
    clk_id == CLOCK_REALTIME || clk_id == CLOCK_MONOTONIC ||
    clk_id == CLOCK_PROCESS_CPUTIME_ID || clk_id == CLOCK_THREAD_CPUTIME_ID ||
    clk_id == CLOCK_MONOTONIC_RAW || clk_id == CLOCK_REALTIME_COARSE ||
    clk_id == CLOCK_MONOTONIC_COARSE || clk_id == CLOCK_BOOTTIME;

  if (!valid_clock)
    return -1;

  if (res != (struct timespec *)0)
  {
    /* Typical resolution is 1ns for high-res clocks, 1ms for coarse */
    res->tv_sec = 0;
    if (clk_id == CLOCK_REALTIME_COARSE || clk_id == CLOCK_MONOTONIC_COARSE)
      res->tv_nsec = 1000000L; /* 1ms */
    else
      res->tv_nsec = 1L; /* 1ns */
  }
  return 0;
}

/* ── clock_gettime (POSIX) ── */
int clock_gettime(clockid_t clk_id, struct timespec *tp)
{
__ESBMC_HIDE:;
  __ESBMC_assert(tp != NULL, "clock_gettime tp must not be NULL");

  /* clk_id validation */
  _Bool valid_clock =
    clk_id == CLOCK_REALTIME || clk_id == CLOCK_MONOTONIC ||
    clk_id == CLOCK_PROCESS_CPUTIME_ID || clk_id == CLOCK_THREAD_CPUTIME_ID ||
    clk_id == CLOCK_MONOTONIC_RAW || clk_id == CLOCK_REALTIME_COARSE ||
    clk_id == CLOCK_MONOTONIC_COARSE || clk_id == CLOCK_BOOTTIME;

  if (!valid_clock)
    return -1;

  __esbmc_fill_timespec_nondet(tp);
  /* For monotonic clocks, time should be non-negative */
  if (
    clk_id == CLOCK_MONOTONIC || clk_id == CLOCK_MONOTONIC_RAW ||
    clk_id == CLOCK_MONOTONIC_COARSE || clk_id == CLOCK_BOOTTIME)
  {
    __ESBMC_assume(tp->tv_sec >= 0);
  }
  return 0;
}

/* ── clock_settime (POSIX) ── */
int clock_settime(clockid_t clk_id, const struct timespec *tp)
{
__ESBMC_HIDE:;
  __ESBMC_assert(tp != NULL, "clock_settime tp must not be NULL");

  /* Only CLOCK_REALTIME can be set (requires privileges) */
  if (clk_id != CLOCK_REALTIME)
    return -1;

  /* Validate timespec */
  if (tp->tv_nsec < 0 || tp->tv_nsec > 999999999L)
    return -1;

  /* Nondeterministically succeed or fail (may lack permissions) */
  return nondet_bool() ? 0 : -1;
}

/* ── clock_nanosleep (POSIX) ── */
int clock_nanosleep(
  clockid_t clk_id,
  int flags,
  const struct timespec *request,
  struct timespec *remain)
{
__ESBMC_HIDE:;
  __ESBMC_assert(request != NULL, "clock_nanosleep request must not be NULL");

  /* clk_id validation (CPU time clocks not allowed) */
  _Bool valid_clock = clk_id == CLOCK_REALTIME || clk_id == CLOCK_MONOTONIC ||
                      clk_id == CLOCK_BOOTTIME;

  if (!valid_clock)
    return 22; /* EINVAL */

  /* Validate timespec */
  if (request->tv_nsec < 0 || request->tv_nsec > 999999999L)
    return 22; /* EINVAL */

  /* For relative sleep, request must be non-negative */
  if (!(flags & TIMER_ABSTIME) && request->tv_sec < 0)
    return 22; /* EINVAL */

  /* Nondeterministically: complete successfully or be interrupted */
  _Bool interrupted = nondet_bool();
  if (interrupted && remain != (struct timespec *)0 && !(flags & TIMER_ABSTIME))
  {
    /* Set remaining time to some value less than requested */
    remain->tv_sec = nondet_long();
    remain->tv_nsec = nondet_long();
    __ESBMC_assume(remain->tv_sec >= 0);
    __ESBMC_assume(remain->tv_nsec >= 0 && remain->tv_nsec <= 999999999L);
    return 4; /* EINTR */
  }
  return 0;
}

/* ── nanosleep (POSIX) ── */
int nanosleep(const struct timespec *req, struct timespec *rem)
{
__ESBMC_HIDE:;
  __ESBMC_assert(req != NULL, "nanosleep req must not be NULL");

  /* Validate timespec */
  if (req->tv_sec < 0 || req->tv_nsec < 0 || req->tv_nsec > 999999999L)
    return -1;

  /* Nondeterministically: complete successfully or be interrupted */
  _Bool interrupted = nondet_bool();
  if (interrupted && rem != (struct timespec *)0)
  {
    /* Set remaining time to some value less than requested */
    rem->tv_sec = nondet_long();
    rem->tv_nsec = nondet_long();
    __ESBMC_assume(rem->tv_sec >= 0);
    __ESBMC_assume(rem->tv_nsec >= 0 && rem->tv_nsec <= 999999999L);
    return -1;
  }
  return 0;
}

/* ── POSIX timezone variables ── */
int daylight = 0;
long timezone = 0;
char *tzname[2] = {0, 0};

/* ── getdate error variable ── */
int getdate_err = 0;

/* ── Static buffer for getdate ── */
static struct tm __esbmc_getdate_buf;

/* ── timer_create (POSIX) ── */
int timer_create(clockid_t clockid, struct sigevent *sevp, timer_t *timerid)
{
__ESBMC_HIDE:;
  __ESBMC_assert(timerid != NULL, "timer_create timerid must not be NULL");

  /* clk_id validation */
  _Bool valid_clock = clockid == CLOCK_REALTIME || clockid == CLOCK_MONOTONIC ||
                      clockid == CLOCK_PROCESS_CPUTIME_ID ||
                      clockid == CLOCK_THREAD_CPUTIME_ID;

  if (!valid_clock)
    return -1;

  /* Nondeterministically succeed or fail */
  if (nondet_bool())
  {
    *timerid = (timer_t)nondet_ulong();
    return 0;
  }
  return -1;
}

/* ── timer_delete (POSIX) ── */
int timer_delete(timer_t timerid)
{
__ESBMC_HIDE:;
  /* Nondeterministically succeed or fail (invalid timer ID) */
  return nondet_bool() ? 0 : -1;
}

/* ── timer_settime (POSIX) ── */
int timer_settime(
  timer_t timerid,
  int flags,
  const struct itimerspec *new_value,
  struct itimerspec *old_value)
{
__ESBMC_HIDE:;
  __ESBMC_assert(new_value != NULL, "timer_settime new_value must not be NULL");

  /* Validate timespec values */
  if (
    new_value->it_value.tv_nsec < 0 ||
    new_value->it_value.tv_nsec > 999999999L ||
    new_value->it_interval.tv_nsec < 0 ||
    new_value->it_interval.tv_nsec > 999999999L)
  {
    return -1;
  }

  /* If old_value is provided, fill with nondet values */
  if (old_value != (struct itimerspec *)0)
  {
    __esbmc_fill_timespec_nondet(&old_value->it_value);
    __esbmc_fill_timespec_nondet(&old_value->it_interval);
  }

  /* Nondeterministically succeed or fail */
  return nondet_bool() ? 0 : -1;
}

/* ── timer_gettime (POSIX) ── */
int timer_gettime(timer_t timerid, struct itimerspec *curr_value)
{
__ESBMC_HIDE:;
  __ESBMC_assert(
    curr_value != NULL, "timer_gettime curr_value must not be NULL");

  /* Fill with nondet values representing remaining time */
  __esbmc_fill_timespec_nondet(&curr_value->it_value);
  __esbmc_fill_timespec_nondet(&curr_value->it_interval);

  /* Nondeterministically succeed or fail */
  return nondet_bool() ? 0 : -1;
}

/* ── timer_getoverrun (POSIX) ── */
int timer_getoverrun(timer_t timerid)
{
__ESBMC_HIDE:;
  /* Return nondet overrun count (0 or positive), or -1 on error */
  _Bool success = nondet_bool();
  if (success)
  {
    int overrun = nondet_int();
    __ESBMC_assume(overrun >= 0);
    return overrun;
  }
  return -1;
}

/* ── clock_getcpuclockid (POSIX) ── */
int clock_getcpuclockid(pid_t pid, clockid_t *clock_id)
{
__ESBMC_HIDE:;
  __ESBMC_assert(
    clock_id != NULL, "clock_getcpuclockid clock_id must not be NULL");

  /* pid 0 means current process - always valid */
  /* Other PIDs may not exist */
  if (pid == 0 || nondet_bool())
  {
    /* Return a CPU-time clock ID for the process */
    *clock_id = CLOCK_PROCESS_CPUTIME_ID;
    return 0;
  }
  return 3; /* ESRCH - no such process */
}

/* ── tzset (POSIX) ── */
void tzset(void)
{
__ESBMC_HIDE:;
  /* Fill timezone variables with nondet values */
  daylight = nondet_int();
  __ESBMC_assume(daylight == 0 || daylight == 1);
  timezone = nondet_long();
  /* tzname pointers remain unchanged (implementation-defined static strings) */
}

/* ── getdate (POSIX XSI) ── */
struct tm *getdate(const char *string)
{
__ESBMC_HIDE:;
  __ESBMC_assert(string != NULL, "getdate string must not be NULL");

  /* Nondeterministically succeed or fail */
  if (nondet_bool())
  {
    __esbmc_fill_tm_nondet(&__esbmc_getdate_buf);
    return &__esbmc_getdate_buf;
  }

  /* Set error code (1-8 are valid POSIX error codes for getdate) */
  getdate_err = nondet_int();
  __ESBMC_assume(getdate_err >= 1 && getdate_err <= 8);
  return (struct tm *)0;
}
