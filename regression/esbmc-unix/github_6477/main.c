/* Re-locking a PTHREAD_MUTEX_RECURSIVE mutex from its owner is legal and does
   not block, so this is not a deadlock. */
#include <pthread.h>

int main(void)
{
  pthread_mutex_t m;
  pthread_mutexattr_t at;
  pthread_mutexattr_init(&at);
  pthread_mutexattr_settype(&at, PTHREAD_MUTEX_RECURSIVE);
  pthread_mutex_init(&m, &at);
  pthread_mutex_lock(&m);
  pthread_mutex_lock(&m);
  pthread_mutex_unlock(&m);
  pthread_mutex_unlock(&m);
  pthread_mutex_destroy(&m);
  pthread_mutexattr_destroy(&at);
  return 0;
}
