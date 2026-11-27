/* #10990: the two threads can deadlock, and main waits for one of them. */
#include <pthread.h>
pthread_mutex_t a = PTHREAD_MUTEX_INITIALIZER, b = PTHREAD_MUTEX_INITIALIZER;
void *ab(void *arg)
{
  pthread_mutex_lock(&a);
  pthread_mutex_lock(&b);
  pthread_mutex_unlock(&b);
  pthread_mutex_unlock(&a);
  return 0;
}
void *ba(void *arg)
{
  pthread_mutex_lock(&b);
  pthread_mutex_lock(&a);
  pthread_mutex_unlock(&a);
  pthread_mutex_unlock(&b);
  return 0;
}
int main()
{
  pthread_t x, y;
  pthread_create(&x, 0, ab, 0);
  pthread_create(&y, 0, ba, 0);
  pthread_join(x, 0);
  return 0;
}
