#include <pthread.h>

void *malloc(unsigned size);
void free(void *p);

pthread_mutex_t *mutex;

void *thread1(void *arg)
{
  pthread_mutex_lock(mutex); 
  pthread_mutex_unlock(mutex);

  return NULL;
}


void *thread2(void *arg)
{
  pthread_mutex_lock(mutex); 
  pthread_mutex_unlock(mutex);

  return NULL;
}


int main(void)
{
  pthread_t id1, id2;

  mutex = (pthread_mutex_t *) malloc(sizeof(pthread_mutex_t));
  pthread_mutex_init(mutex, NULL);

  pthread_create(&id1, NULL, thread1, NULL);
  pthread_create(&id2, NULL, thread2, NULL);

  pthread_join(id1, NULL);
  pthread_join(id2, NULL);

  free(mutex);

  free(mutex);

  return 0;
}
