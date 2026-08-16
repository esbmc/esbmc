/* The barrier orders the write in the worker before the read in main. */
#include <pthread.h>
#include <assert.h>
pthread_barrier_t bar;
int shared = 0;

void *t(void *p)
{
  shared = 42;
  pthread_barrier_wait(&bar);
  return 0;
}

int main(void)
{
  pthread_t a;
  pthread_barrier_init(&bar, 0, 2);
  pthread_create(&a, 0, t, 0);
  pthread_barrier_wait(&bar);
  assert(shared == 42);
  pthread_join(a, 0);
  return 0;
}
