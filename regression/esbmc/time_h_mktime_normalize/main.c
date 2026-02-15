#include <time.h>
#include <assert.h>

int main()
{
  struct tm t;
  t.tm_year = 2024 - 1900;
  t.tm_mon = 13;
  t.tm_mday = 1;
  t.tm_hour = 0;
  t.tm_min = 0;
  t.tm_sec = 0;
  t.tm_isdst = -1;
  time_t result = mktime(&t);
  if (result != (time_t)(-1))
  {
    assert(t.tm_mon >= 0 && t.tm_mon <= 11);
  }
  return 0;
}
