/* #10990: under --partial-order --program-only a lock and an unlock are
   events on their mutex. */
#include <pthread.h>
int x = 0;
pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;
void *t(void *a)
{
  pthread_mutex_lock(&m);
  x = 1;
  pthread_mutex_unlock(&m);
  return 0;
}
int main()
{
  pthread_t a;
  pthread_create(&a, 0, t, 0);
  return 0;
}
