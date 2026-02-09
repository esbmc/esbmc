#include <time.h>
#include <assert.h>

int main()
{
  struct timespec ts;
  int ret = clock_gettime(CLOCK_REALTIME, &ts);

  /* clock_gettime should succeed for CLOCK_REALTIME */
  assert(ret == 0);

  /* nanoseconds must be in valid range */
  assert(ts.tv_nsec >= 0 && ts.tv_nsec <= 999999999L);

  return 0;
}
