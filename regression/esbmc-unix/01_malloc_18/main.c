#include <pthread.h>
#include <stdlib.h>

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

  if (mutex==NULL)
    exit(0);

  pthread_mutex_init(mutex, NULL);

  pthread_create(&id1, NULL, thread1, NULL);
  pthread_create(&id2, NULL, thread2, NULL);

  free(mutex);

  return 0;
}
