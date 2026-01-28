#include <time.h>
#include <assert.h>

int main()
{
  time_t t = time(NULL);
  struct tm *r = gmtime(&t);
  if (r != (void *)0)
  {
    assert(r->tm_sec >= 0 && r->tm_sec <= 60);
    assert(r->tm_min >= 0 && r->tm_min <= 59);
    assert(r->tm_hour >= 0 && r->tm_hour <= 23);
    assert(r->tm_mday >= 1 && r->tm_mday <= 31);
    assert(r->tm_mon >= 0 && r->tm_mon <= 11);
    assert(r->tm_wday >= 0 && r->tm_wday <= 6);
    assert(r->tm_yday >= 0 && r->tm_yday <= 365);
  }
  return 0;
}
