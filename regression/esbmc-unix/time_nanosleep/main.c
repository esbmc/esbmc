#include <time.h>
#include <assert.h>

int main()
{
  struct timespec req = {0, 1000}; /* 1 microsecond */
  struct timespec rem;

  int ret = nanosleep(&req, &rem);

  /* nanosleep returns 0 on success or -1 if interrupted */
  assert(ret == 0 || ret == -1);

  /* If interrupted, remaining time must be valid */
  if (ret == -1)
  {
    assert(rem.tv_sec >= 0);
    assert(rem.tv_nsec >= 0 && rem.tv_nsec <= 999999999L);
  }

  return 0;
}
