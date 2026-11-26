/* #10990: with --deadlock-check, a wait that no signal ends is a deadlock, but
   --partial-order lets every wait wake at will. */
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
