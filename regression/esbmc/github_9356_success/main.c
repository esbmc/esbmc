#include <assert.h>
#include <string.h>
#include <time.h>

/* #9356: asctime, ctime and their _r forms return a 25-character string
 * ending in a newline (C17 7.27.3.1). */
int main(void)
{
  struct tm t = {0};
  t.tm_year = 100;
  t.tm_mday = 1;
  t.tm_wday = 6;
  time_t now = 0;
  char buf[26];
  assert(strlen(asctime(&t)) == 25);
  assert(strlen(ctime(&now)) == 25);
  assert(strlen(asctime_r(&t, buf)) == 25);
  assert(buf[24] == '\n' && buf[13] == ':');
  assert(strlen(ctime_r(&now, buf)) == 25);
  return 0;
}
