#include <time.h>
#include <assert.h>

int main()
{
  struct tm tm = {0};
  tm.tm_year = 124; /* 2024 */
  tm.tm_mon = 0;    /* January */
  tm.tm_mday = 15;
  tm.tm_hour = 12;
  tm.tm_min = 30;
  tm.tm_sec = 45;

  char buf[26];
  char *ret = asctime_r(&tm, buf);

  /* asctime_r should return the buffer on success */
  assert(ret == buf);
  /* Buffer should be null-terminated */
  assert(buf[25] == '\0');

  return 0;
}
