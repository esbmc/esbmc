#include <time.h>
#include <assert.h>

int main()
{
  assert(sizeof(time_t) >= sizeof(int));
  assert(sizeof(clock_t) > 0);

  struct tm t;
  (void)t.tm_sec;
  (void)t.tm_min;
  (void)t.tm_hour;
  (void)t.tm_mday;
  (void)t.tm_mon;
  (void)t.tm_year;
  (void)t.tm_wday;
  (void)t.tm_yday;
  (void)t.tm_isdst;

  struct timespec ts;
  (void)ts.tv_sec;
  (void)ts.tv_nsec;

  return 0;
}
