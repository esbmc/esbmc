#include <assert.h>
#include <pthread.h>

pthread_barrier_t a, b;

void *t1(void *arg)
{
  pthread_barrier_wait(&b);
  pthread_barrier_wait(&a);
  return 0;
}

int main(void)
{
  pthread_t t;
  pthread_barrier_init(&a, 0, 2);
  pthread_barrier_init(&b, 0, 2);
  pthread_create(&t, 0, t1, 0);
  pthread_barrier_wait(&a);
  pthread_barrier_wait(&b);
  assert(0);
  return 0;
}
