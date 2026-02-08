#include <time.h>
#include <assert.h>

int main()
{
  struct timespec res;

  /* CLOCK_REALTIME should succeed */
  int ret = clock_getres(CLOCK_REALTIME, &res);
  assert(ret == 0);
  assert(res.tv_sec == 0);
  assert(res.tv_nsec == 1L); /* High-res clock has 1ns resolution */

  /* CLOCK_REALTIME_COARSE should have 1ms resolution */
  ret = clock_getres(CLOCK_REALTIME_COARSE, &res);
  assert(ret == 0);
  assert(res.tv_nsec == 1000000L);

  /* Invalid clock should fail */
  ret = clock_getres(999, &res);
  assert(ret == -1);

  return 0;
}
