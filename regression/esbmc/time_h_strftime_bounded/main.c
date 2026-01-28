#include <time.h>
#include <assert.h>

int main()
{
  struct tm t;
  t.tm_year = 2024 - 1900;
  t.tm_mon = 5;
  t.tm_mday = 15;
  t.tm_hour = 10;
  t.tm_min = 30;
  t.tm_sec = 0;
  t.tm_wday = 6;
  t.tm_yday = 166;
  t.tm_isdst = 1;
  char buf[64];
  size_t result = strftime(buf, sizeof(buf), "%Y", &t);
  assert(result == 0 || result <= sizeof(buf) - 1);
  return 0;
}
