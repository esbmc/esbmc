// Issue #8220, race-free variant: both threads take the mutex.
#include <pthread.h>

int global = 0;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_t id1, id2;

void *t(void *arg)
{
  pthread_mutex_lock(&mutex);
  global++;
  pthread_mutex_unlock(&mutex);
  return NULL;
}

int main(void)
{
  pthread_create(&id1, NULL, t, NULL);
  pthread_create(&id2, NULL, t, NULL);
  pthread_join(id1, NULL);
  pthread_join(id2, NULL);
  return 0;
}
