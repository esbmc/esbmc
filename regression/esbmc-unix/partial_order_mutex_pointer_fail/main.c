/* #10990: two mutexes do not exclude each other. */
#include <pthread.h>
#include <assert.h>
int x = 0;
pthread_mutex_t m1 = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t m2 = PTHREAD_MUTEX_INITIALIZER;
void increment(pthread_mutex_t *l)
{
  pthread_mutex_lock(l);
  int tmp = x;
  x = tmp + 1;
  pthread_mutex_unlock(l);
}
void *t1(void *a)
{
  increment(&m1);
  return 0;
}
void *t2(void *a)
{
  increment(&m2);
  return 0;
}
int main()
{
  pthread_t a, b;
  pthread_create(&a, 0, t1, 0);
  pthread_create(&b, 0, t2, 0);
  pthread_join(a, 0);
  pthread_join(b, 0);
  assert(x == 2);
  return 0;
}
