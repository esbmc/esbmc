#include <time.h>
#include <assert.h>

int main()
{
  /* Invalid nanoseconds - should fail */
  struct timespec req = {0, 1000000000L}; /* 1 billion ns = 1 second, but invalid */

  int ret = nanosleep(&req, (struct timespec *)0);
  assert(ret == 0); /* Should fail - invalid tv_nsec returns -1 */

  return 0;
}
