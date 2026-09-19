/* main writes a local by name after handing its address to the worker. The
 * write must be a context-switch point, or no schedule lets the worker read
 * the local after it. */
#include <assert.h>
#include <pthread.h>

void *worker(void *arg)
{
  assert(*(int *)arg == 0 || *(int *)arg == 1);
  return 0;
}

int main()
{
  int done = 0;
  pthread_t t;
  pthread_create(&t, 0, worker, &done);
  done = 1;
  pthread_join(t, 0);
  return 0;
}
