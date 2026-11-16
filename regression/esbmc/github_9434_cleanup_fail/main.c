// #9434: a cleanup handler run by pthread_exit is application code; main can
// observe its intermediate state while the exiting thread runs it.
#include <assert.h>
#include <pthread.h>

int x = 0;

void handler(void *p)
{
  x = 1;
  x = 0;
}

void *t(void *a)
{
  pthread_cleanup_push(handler, 0);
  pthread_exit(0);
  pthread_cleanup_pop(0);
  return 0;
}

int main(void)
{
  pthread_t th;
  pthread_create(&th, 0, t, 0);
  assert(x != 1);
  pthread_join(th, 0);
  return 0;
}
