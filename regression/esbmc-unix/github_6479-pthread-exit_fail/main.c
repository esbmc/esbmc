/* main hands the process over to its threads with pthread_exit, so the worker
   really does outlive main and really does deadlock on the lock main holds. */
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
  pthread_exit(0);
}
