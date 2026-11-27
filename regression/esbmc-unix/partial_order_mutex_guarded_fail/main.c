/* #10990: on the path that releases the mutex, the thread writes back the
   value it read before, so the other thread's increment can be lost. */
#include <pthread.h>
#include <assert.h>
int nondet_int(void);
int x = 0;
pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;
void *t(void *a)
{
  pthread_mutex_lock(&m);
  int tmp = x;
  if (nondet_int())
  {
    pthread_mutex_unlock(&m);
    pthread_mutex_lock(&m);
  }
  x = tmp + 1;
  pthread_mutex_unlock(&m);
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
