/* #10789: a field edited in gmtime's buffer must reach timegm's result. */
#include <assert.h>
#include <time.h>

int main(void)
{
  time_t t0 = 1000;
  struct tm *p = gmtime(&t0);
  p->tm_sec += 1;
  assert(timegm(p) == 1000);
  return 0;
}
