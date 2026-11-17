/* #10789: timegm normalised only tm_mon, so tm_sec = -1 never borrowed a day
   and tm_mday kept the value written before the call. */
#include <assert.h>
#include <time.h>

int main(void)
{
  struct tm t = {0};
  t.tm_year = 2024 - 1900;
  t.tm_mday = 9;
  t.tm_sec = -1;

  timegm(&t);

  assert(t.tm_mday == 9);
  return 0;
}
