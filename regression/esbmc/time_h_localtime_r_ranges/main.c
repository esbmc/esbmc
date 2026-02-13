#include <time.h>
#include <assert.h>

int main()
{
  time_t t = time(NULL);
  struct tm buf;
  struct tm *r = localtime_r(&t, &buf);
  if (r != (void *)0)
  {
    assert(r == &buf);
    assert(buf.tm_sec >= 0 && buf.tm_sec <= 60);
    assert(buf.tm_min >= 0 && buf.tm_min <= 59);
    assert(buf.tm_hour >= 0 && buf.tm_hour <= 23);
    assert(buf.tm_mday >= 1 && buf.tm_mday <= 31);
    assert(buf.tm_mon >= 0 && buf.tm_mon <= 11);
    assert(buf.tm_wday >= 0 && buf.tm_wday <= 6);
    assert(buf.tm_yday >= 0 && buf.tm_yday <= 365);
  }
  return 0;
}
