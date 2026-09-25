#include <time.h>
#include <assert.h>

int main(void)
{
  struct tm t = {0};
  t.tm_year = 102;
  t.tm_mon = 21; /* out of range: carries one year */
  t.tm_mday = 9;
  t.tm_isdst = 1; /* UTC has no DST; timegm clears this */

  timegm(&t);

  assert(t.tm_year == 103);
  assert(t.tm_mon == 9);
  assert(t.tm_mday == 9);
  assert(t.tm_isdst == 0);

  /* timegm is gmtime's inverse. */
  time_t x = 1034000000;
  struct tm *g = gmtime(&x);
  assert(timegm(g) == x);

  return 0;
}
