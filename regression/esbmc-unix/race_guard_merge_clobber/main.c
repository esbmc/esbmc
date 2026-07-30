// Minimal form of the defect behind race_guard_self_clear: a branch on a
// global whose body writes that same global loses a concurrent write to it
// across the merge, so `receive` is wrongly proved to be 0 here.
#include <assert.h>
#include <pthread.h>

_Bool receive = 0;

void *t1(void *arg)
{
  if (receive)
    receive = 0;
  assert(!receive);
  return NULL;
}

int main()
{
  pthread_t id;
  pthread_create(&id, NULL, t1, NULL);
  receive = 1;
  return 0;
}
