/* The threads wait on the barriers in opposite orders when top != 0, so that
   path deadlocks and never reaches the assertion. */
#include <assert.h>
#include <pthread.h>

pthread_barrier_t b1, b2;

void *worker(void *arg)
{
  pthread_barrier_wait(&b1);
  pthread_barrier_wait(&b2);
  return NULL;
}

int main(void)
{
  int top = __VERIFIER_nondet_int();
  int i = 0;
  pthread_barrier_init(&b1, NULL, 2);
  pthread_barrier_init(&b2, NULL, 2);
  pthread_t t;
  pthread_create(&t, NULL, worker, NULL);
  if (top)
  {
    pthread_barrier_wait(&b2);
    pthread_barrier_wait(&b1);
    i = 2;
  }
  assert(i == 0);
  return 0;
}
