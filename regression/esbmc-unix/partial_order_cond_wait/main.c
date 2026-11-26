/* #10990: a condition wait takes its mutex back before it returns, so the
   other thread's write cannot fall between the wait and the assertion. */
#include <pthread.h>
#include <assert.h>
int x = 0;
pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t c = PTHREAD_COND_INITIALIZER;
void *t(void *a)
{
  pthread_mutex_lock(&m);
  x = 2;
  pthread_cond_broadcast(&c);
  pthread_mutex_unlock(&m);
  return 0;
}
int main()
{
  pthread_t id;
  pthread_create(&id, 0, t, 0);
  pthread_mutex_lock(&m);
  pthread_cond_wait(&c, &m);
  x = 1;
  assert(x == 1);
  pthread_mutex_unlock(&m);
  return 0;
}
