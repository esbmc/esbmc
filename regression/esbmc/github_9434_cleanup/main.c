// #9434: pthread_exit runs every pushed cleanup handler, last pushed first,
// before pthread_join returns.
#include <assert.h>
#include <pthread.h>

int order = 0;

void first(void *p)
{
  order = order * 10 + 1;
}

void second(void *p)
{
  order = order * 10 + 2;
}

void *t(void *a)
{
  pthread_cleanup_push(first, 0);
  pthread_cleanup_push(second, 0);
  pthread_exit(0);
  pthread_cleanup_pop(0);
  pthread_cleanup_pop(0);
  return 0;
}

int main(void)
{
  pthread_t th;
  pthread_create(&th, 0, t, 0);
  pthread_join(th, 0);
  assert(order == 21);
  return 0;
}
