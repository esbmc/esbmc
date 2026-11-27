#include <pthread.h>
#include <assert.h>
#include <time.h>
int x;
pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t c = PTHREAD_COND_INITIALIZER;
struct timespec ts;
int main()
{
  pthread_mutex_lock(&m);
  pthread_cond_timedwait(&c, &m, &ts);
  pthread_mutex_unlock(&m);
  assert(x == 0);
  return 0;
}
