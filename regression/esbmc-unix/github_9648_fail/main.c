/* Two parties reach a barrier of two, so it opens and the assertion fails. */
#include <assert.h>
#include <pthread.h>

pthread_barrier_t b;

void *worker(void *arg)
{
  pthread_barrier_wait(&b);
  return NULL;
}

int main(void)
{
  int i = 0;
  pthread_barrier_init(&b, NULL, 2);
  pthread_t t;
  pthread_create(&t, NULL, worker, NULL);
  pthread_barrier_wait(&b);
  i = 1;
  assert(i == 0);
  return 0;
}
