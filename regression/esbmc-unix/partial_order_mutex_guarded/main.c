/* #10990: an unlock on one path releases the mutex only on that path, and
   the thread reads x again once it holds the mutex back. */
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
    tmp = x;
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
