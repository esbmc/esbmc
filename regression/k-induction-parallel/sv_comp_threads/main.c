/* --sv-comp sends a program that starts a thread to incremental BMC, also when
   --k-induction-parallel is given (#10293). */
#include <pthread.h>
#include <assert.h>
int x;
pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;
void *t(void *arg)
{
  pthread_mutex_lock(&m);
  x++;
  pthread_mutex_unlock(&m);
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
