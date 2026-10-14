/* A mutex in main's frame, reached by the worker through the pthread_create
 * argument: MPOR must key both threads' accesses to it, or the schedule where
 * the worker takes the lock first is pruned. */
#include <assert.h>
#include <pthread.h>

struct shared
{
  pthread_mutex_t m;
  int done;
};

void *worker(void *arg)
{
  struct shared *s = arg;
  pthread_mutex_lock(&s->m);
  s->done = 1;
  pthread_mutex_unlock(&s->m);
  return 0;
}

int main()
{
  struct shared s = {PTHREAD_MUTEX_INITIALIZER, 0};
  pthread_t t;
  pthread_create(&t, 0, worker, &s);
  pthread_mutex_lock(&s.m);
  assert(s.done == 0 || s.done == 1);
  pthread_mutex_unlock(&s.m);
  return 0;
}
