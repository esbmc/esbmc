#include <pthread.h>

int g;
pthread_mutex_t mutex;

void *thread1(void *arg)
{
  pthread_mutex_lock(&mutex); 
  g = 1;
  pthread_mutex_unlock(&mutex);
}

void *thread2(void *arg)
{
  pthread_mutex_lock(&mutex);
  g=0;
  assert(g == 0);
  pthread_mutex_unlock(&mutex);
}

int main(void)
{
  pthread_t id1, id2;

  pthread_mutex_init(&mutex, NULL);
  pthread_create(&id1, NULL, thread1, NULL);
  pthread_create(&id2, NULL, thread2, NULL);
  pthread_mutex_destroy(&mutex);

  return 0;
}
