/* #10990: under --partial-order --program-only a condition wait releases its
   mutex and then takes it back. */
#include <pthread.h>
pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t c = PTHREAD_COND_INITIALIZER;
int main()
{
  pthread_mutex_lock(&m);
  pthread_cond_wait(&c, &m);
  pthread_mutex_unlock(&m);
  return 0;
}
