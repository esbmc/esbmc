#pragma once

/* POSIX <sys/time.h>. Without a copy here, a program including it falls
 * through to the platform SDK, whose header pulls <sys/_types/_fd_def.h> and
 * <sys/_types/_timeval.h>. Those redefine the fd_set and timeval that
 * <sys/select.h> already defines, so on macOS such a program did not compile
 * at all. */
#include <sys/select.h> /* fd_set, struct timeval, select */
#include <time.h>       /* struct timespec, time_t */

__ESBMC_C_CPP_BEGIN

struct timezone
{
  int tz_minuteswest; // cppcheck-suppress unusedStructMember
  int tz_dsttime;     // cppcheck-suppress unusedStructMember
};

int gettimeofday(struct timeval *tv, struct timezone *tz);
int settimeofday(const struct timeval *tv, const struct timezone *tz);

#define timerclear(tvp) ((tvp)->tv_sec = (tvp)->tv_usec = 0)
#define timerisset(tvp) ((tvp)->tv_sec || (tvp)->tv_usec)
#define timercmp(a, b, CMP)                                                    \
  (((a)->tv_sec == (b)->tv_sec) ? ((a)->tv_usec CMP(b)->tv_usec)               \
                                : ((a)->tv_sec CMP(b)->tv_sec))

__ESBMC_C_CPP_END
