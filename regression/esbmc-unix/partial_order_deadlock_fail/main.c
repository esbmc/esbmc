/* #10990: each thread can hold the mutex the other waits for. */
#include <pthread.h>
pthread_mutex_t a = PTHREAD_MUTEX_INITIALIZER, b = PTHREAD_MUTEX_INITIALIZER;
void *t(void *arg)
{
  pthread_mutex_lock(&b);
  pthread_mutex_lock(&a);
  pthread_mutex_unlock(&a);
  pthread_mutex_unlock(&b);
  return 0;
}
int main()
{
  pthread_t id;
  pthread_create(&id, 0, t, 0);
  pthread_mutex_lock(&a);
  pthread_mutex_lock(&b);
  pthread_mutex_unlock(&b);
  pthread_mutex_unlock(&a);
  pthread_join(id, 0);
  return 0;
}
