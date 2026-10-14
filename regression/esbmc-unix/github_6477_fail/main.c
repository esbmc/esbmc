/* A recursive mutex stays held until the outermost unlock: the worker can
   never acquire it, so the join really does deadlock. */
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
  pthread_mutexattr_t at;
  pthread_mutexattr_init(&at);
  pthread_mutexattr_settype(&at, PTHREAD_MUTEX_RECURSIVE);
  pthread_mutex_init(&m, &at);
  pthread_mutex_lock(&m);
  pthread_mutex_lock(&m);
  pthread_create(&a, 0, w, 0);
  pthread_mutex_unlock(&m); /* still held at the outer level */
  pthread_join(a, 0);
  return 0;
}
