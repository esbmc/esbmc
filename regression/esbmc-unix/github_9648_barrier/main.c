/* #9648: when top != 0 the two threads wait on the barriers in opposite
 * orders and deadlock, so the assertion is reached only when top == 0. */
#include <assert.h>
#include <pthread.h>

int nondet_int(void);
pthread_barrier_t b1, b2;
int top;

void *t1(void *arg)
{
  if (top)
  {
    pthread_barrier_wait(&b2);
    pthread_barrier_wait(&b1);
  }
  else
  {
    pthread_barrier_wait(&b1);
    pthread_barrier_wait(&b2);
  }
  return 0;
}

int main(void)
{
  pthread_t t;
  top = nondet_int();
  pthread_barrier_init(&b1, 0, 2);
  pthread_barrier_init(&b2, 0, 2);
  pthread_create(&t, 0, t1, 0);
  pthread_barrier_wait(&b1);
  pthread_barrier_wait(&b2);
  assert(top == 0);
  return 0;
}
