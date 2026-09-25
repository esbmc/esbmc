/* main returns while holding m.  Returning from main calls exit(), the process
   terminates, and the worker is torn down -- this is NOT a deadlock. */
#include <pthread.h>
pthread_mutex_t m;

void *w(void *p)
{
  pthread_mutex_lock(&m);
  pthread_mutex_unlock(&m);
  return 0;
}

int main(void)
{
  pthread_t a;
  pthread_mutex_init(&m, 0);
  pthread_mutex_lock(&m);
  pthread_create(&a, 0, w, 0);
  return 0;
}
