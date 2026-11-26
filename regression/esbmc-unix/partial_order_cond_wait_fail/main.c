/* #10990: a condition wait releases its mutex, so the other thread's write
   can fall between the write before the wait and the assertion after it. */
#include <pthread.h>
#include <assert.h>
int x = 0;
pthread_mutex_t m;
pthread_cond_t c;
void *t(void *a)
{
  pthread_mutex_lock(&m);
  x = 2;
  pthread_cond_signal(&c);
  pthread_mutex_unlock(&m);
  return 0;
}
int main()
{
  pthread_t id;
  pthread_mutex_init(&m, 0);
  pthread_cond_init(&c, 0);
  pthread_create(&id, 0, t, 0);
  pthread_mutex_lock(&m);
  x = 1;
  pthread_cond_wait(&c, &m);
  assert(x == 1);
  pthread_mutex_unlock(&m);
  pthread_cond_destroy(&c);
  return 0;
}
