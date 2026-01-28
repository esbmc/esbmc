#include <time.h>
#include <assert.h>

int main()
{
  struct timespec ts;
  int result = timespec_get(&ts, TIME_UTC);
  if (result == TIME_UTC)
  {
    assert(ts.tv_nsec >= 0);
    assert(ts.tv_nsec <= 999999999L);
  }
  return 0;
}
