#include <pthread.h>

pthread_mutex_t mutex;

void *thread1(void *arg)
{
  pthread_mutex_lock(&mutex); 
  assert(0);
  pthread_mutex_unlock(&mutex);
}

int main(void)
{
  pthread_t id1;

  pthread_mutex_init(&mutex, NULL);
  pthread_create(&id1, NULL, thread1, NULL);
  pthread_mutex_destroy(&mutex);

  return 0;
}
