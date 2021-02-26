#include <pthread.h>

int g;
pthread_mutex_t mutex;

void *thread1(void *arg)
{
  pthread_mutex_lock(&mutex); 
  g = 1;
  pthread_mutex_unlock(&mutex);
}

int main(void)
{
  pthread_t id1, id2;

  pthread_mutex_init(&mutex, NULL);
  pthread_create(&id1, NULL, thread1, NULL);

  // this may fail
  pthread_mutex_lock(&mutex);
  g=0;
  assert(g == 0);
  pthread_mutex_unlock(&mutex);

  pthread_mutex_destroy(&mutex);

  return 0;
}
