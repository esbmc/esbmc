/* #10990: main waits for a thread that waits for the mutex main holds, so
   no execution runs to the end. */
#include <pthread.h>
pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;
void *t(void *arg)
{
  pthread_mutex_lock(&m);
  pthread_mutex_unlock(&m);
  return 0;
}
int main()
{
  pthread_t id;
  pthread_mutex_lock(&m);
  pthread_create(&id, 0, t, 0);
  pthread_join(id, 0);
  pthread_mutex_unlock(&m);
  return 0;
}
