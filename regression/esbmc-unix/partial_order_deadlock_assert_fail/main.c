/* #10990: with --deadlock-check, an assertion is still checked: the
   increments are not under the mutex, so one can be lost. */
#include <assert.h>
#include <pthread.h>
int x = 0;
pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;
void *t(void *arg)
{
  int tmp = x;
  pthread_mutex_lock(&m);
  pthread_mutex_unlock(&m);
  x = tmp + 1;
  return 0;
}
int main()
{
  pthread_t a, b;
  pthread_create(&a, 0, t, 0);
  pthread_create(&b, 0, t, 0);
  pthread_join(a, 0);
  pthread_join(b, 0);
  assert(x == 2);
  return 0;
}
