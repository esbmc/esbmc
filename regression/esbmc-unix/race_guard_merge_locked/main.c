// race_guard_merge_clobber with the branch and the concurrent write under one
// mutex. The extra context-switch points the branch now generates must not
// invent a schedule that splits the critical section (#6558).
#include <assert.h>
#include <pthread.h>

_Bool receive = 0;
pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;

void *t1(void *arg)
{
  pthread_mutex_lock(&m);
  if (receive)
    receive = 0;
  assert(!receive);
  pthread_mutex_unlock(&m);
  return NULL;
}

int main()
{
  pthread_t id;
  pthread_create(&id, NULL, t1, NULL);
  pthread_mutex_lock(&m);
  receive = 1;
  pthread_mutex_unlock(&m);
  return 0;
}
