/* #10990: t ends holding m. If t locks first, u waits forever at its lock and
   never reaches its assert; otherwise u reads flag before t writes it. */
#include <pthread.h>
#include <assert.h>
int flag = 0;
pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;
void *t(void *a)
{
  pthread_mutex_lock(&m);
  flag = 1;
  return 0;
}
void *u(void *a)
{
  pthread_mutex_lock(&m);
  int f = flag;
  pthread_mutex_unlock(&m);
  assert(f == 0);
  return 0;
}
int main()
{
  pthread_t a, b;
  pthread_create(&a, 0, t, 0);
  pthread_create(&b, 0, u, 0);
  return 0;
}
