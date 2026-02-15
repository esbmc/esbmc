#include <time.h>
#include <assert.h>

int main()
{
  struct tm t;
  t.tm_year = 2024 - 1900;
  t.tm_mon = 0;
  t.tm_mday = 1;
  t.tm_hour = 12;
  t.tm_min = 30;
  t.tm_sec = 45;
  t.tm_wday = 1;
  t.tm_yday = 0;
  t.tm_isdst = 0;
  char buf[1];
  size_t result = strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &t);
  assert(result == 0);
  return 0;
}
