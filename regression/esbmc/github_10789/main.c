/* #10789: timegm normalised only tm_mon, so a negative tm_sec never borrowed
   from the minutes, hours and days above it. */
#include <assert.h>
#include <time.h>

int main(void)
{
  struct tm t = {0};
  t.tm_year = 2024 - 1900;
  t.tm_mon = 0;
  t.tm_mday = 9;
  t.tm_sec = -1;

  time_t r = timegm(&t);

  assert(r == 1704758399);
  assert(t.tm_year == 2024 - 1900);
  assert(t.tm_mon == 0);
  assert(t.tm_mday == 8);
  assert(t.tm_hour == 23);
  assert(t.tm_min == 59);
  assert(t.tm_sec == 59);
  assert(t.tm_wday == 1);
  assert(t.tm_yday == 7);
  return 0;
}
