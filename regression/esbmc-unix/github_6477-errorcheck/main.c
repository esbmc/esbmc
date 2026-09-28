/* PTHREAD_MUTEX_ERRORCHECK reports rather than blocks or traps: EDEADLK for a
   relock by the owner, EPERM for an unlock by a thread that does not hold it. */
#include <pthread.h>
#include <errno.h>
#include <assert.h>

int main(void)
{
  pthread_mutex_t m;
  pthread_mutexattr_t at;
  int kind;
  pthread_mutexattr_init(&at);
  pthread_mutexattr_settype(&at, PTHREAD_MUTEX_ERRORCHECK);
  pthread_mutexattr_gettype(&at, &kind);
  assert(kind == PTHREAD_MUTEX_ERRORCHECK);
  pthread_mutex_init(&m, &at);
  assert(pthread_mutex_lock(&m) == 0);
  assert(pthread_mutex_lock(&m) == EDEADLK);
  assert(pthread_mutex_unlock(&m) == 0);
  assert(pthread_mutex_unlock(&m) == EPERM);
  pthread_mutex_destroy(&m);
  return 0;
}
