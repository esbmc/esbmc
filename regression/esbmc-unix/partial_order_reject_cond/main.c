#include <pthread.h>
#include <assert.h>
int x;
pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t c = PTHREAD_COND_INITIALIZER;
void *t(void *arg)
{
  pthread_mutex_lock(&m);
  x = 1;
  pthread_cond_signal(&c);
  pthread_mutex_unlock(&m);
  return 0;
}
int main()
{
  pthread_t id;
  pthread_create(&id, 0, t, 0);
  assert(x == 0);
  return 0;
}
