/*******************************************************************\

Module: Time Stopping

Author: Daniel Kroening

Date: February 2004

\*******************************************************************/

#if defined(_WIN32) && !defined(__MINGW32__)
#include <windows.h>
#include <winbase.h>
#else
#include <sys/time.h>
#endif

#include <iomanip>
#include <sstream>
#include <util/time_stopping.h>

#if defined(_WIN32) && !defined(__MINGW32__)
struct timezone
{
  int dummy;
};

void gettimeofday(struct timeval *p, struct timezone *tz)
{
  union {
    long long ns100; /*time since 1 Jan 1601 in 100ns units */
    FILETIME ft;
  } _now;

  GetSystemTimeAsFileTime(&(_now.ft));
  p->tv_usec = (long)((_now.ns100 / 10LL) % 1000000LL);
  p->tv_sec = (long)((_now.ns100 - (116444736000000000LL)) / 10000000LL);
  return;
}
#endif

fine_timet current_time()
{
  struct timeval tv;
  struct timezone tz;

  gettimeofday(&tv, &tz);

  return tv.tv_usec / 1000 + (fine_timet)tv.tv_sec * 1000;
}

void output_time(const fine_timet &fine_time, std::ostream &out)
{
  out << std::setiosflags(std::ios::fixed) << std::setprecision(3)
      << (double)(fine_time) / 1000;
}

std::string time2string(const fine_timet &fine_time)
{
  std::ostringstream out;
  output_time(fine_time, out);
  return out.str();
}
