#include <pthread.h>
pthread_cond_t c = PTHREAD_COND_INITIALIZER;
int main()
{
  pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;
  pthread_cond_wait(&c, &m);
  return 0;
}
