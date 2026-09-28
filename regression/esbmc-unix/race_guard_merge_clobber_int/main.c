// As race_guard_merge_clobber, but non-boolean and with a body value that is
// not the guard-falsifying one, so the merged term does not fold to a
// constant. Still wrongly proved: the defect is the guard and the not-taken
// arm naming the same stale value, not the folding.
#include <assert.h>
#include <pthread.h>

int receive = 0;

void *t1(void *arg)
{
  if (receive == 1)
    receive = 5;
  assert(receive != 1);
  return NULL;
}

int main()
{
  pthread_t id;
  pthread_create(&id, NULL, t1, NULL);
  receive = 1;
  return 0;
}
