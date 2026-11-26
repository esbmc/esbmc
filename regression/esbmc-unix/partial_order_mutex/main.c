/* #10990: the section 2 counter, its increment made of a separate read and
   write, kept whole by the mutex. */
#include <pthread.h>
#include <assert.h>
int x = 0;
pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;
void *t(void *a)
{
  pthread_mutex_lock(&m);
  int tmp = x;
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
