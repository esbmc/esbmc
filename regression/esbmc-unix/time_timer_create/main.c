#include <time.h>
#include <assert.h>

int main()
{
  timer_t timerid;

  /* Create a timer with CLOCK_REALTIME */
  int ret = timer_create(CLOCK_REALTIME, (struct sigevent *)0, &timerid);

  /* timer_create returns 0 on success, -1 on error */
  assert(ret == 0 || ret == -1);

  if (ret == 0)
  {
    /* If timer was created, we should be able to delete it */
    int del_ret = timer_delete(timerid);
    assert(del_ret == 0 || del_ret == -1);
  }

  return 0;
}
