/* The lost update is found by incremental BMC, not by the parallel
   k-induction driver (#10293). */
#include <pthread.h>
#include <assert.h>
int x;
pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;
void *t(void *arg)
{

  int tmp = x;
  x = tmp + 1;

  return 0;
}
int main(void)
{
  pthread_t a, b;
  pthread_create(&a, 0, t, 0);
  pthread_create(&b, 0, t, 0);
  pthread_join(a, 0);
  pthread_join(b, 0);
  assert(x == 2);
  return 0;
}
