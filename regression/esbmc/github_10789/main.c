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

  /* Dates before 1970, a negative tm_mon, and tm_yday after a leap day:
     tm_mon = -10 in 1970 is March 1969. */
  struct tm u = {0};
  u.tm_year = 1968 - 1900;
  u.tm_mon = 2;
  u.tm_mday = 1;
  assert(timegm(&u) == -57974400);
  assert(u.tm_yday == 60);
  assert(u.tm_wday == 5);

  struct tm v = {0};
  v.tm_year = 1970 - 1900;
  v.tm_mon = -10;
  v.tm_mday = 1;
  timegm(&v);
  assert(v.tm_year == 1969 - 1900 && v.tm_mon == 2 && v.tm_yday == 59);
  return 0;
}
