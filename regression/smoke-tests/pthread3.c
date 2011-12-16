#include <pthread.h>

int g;
pthread_mutex_t mutex=PTHREAD_MUTEX_INITIALIZER;

void *t1(void *arg)
{
  pthread_mutex_lock(&mutex);
  g=1;              
  g=0;        
  pthread_mutex_unlock(&mutex);
}

void *t2(void *arg)
{
  pthread_mutex_lock(&mutex);
  assert(g==0);              
  pthread_mutex_unlock(&mutex);
}

int main()
{
  pthread_t id1, id2;

  pthread_create(&id1, NULL, t1, NULL);
  pthread_create(&id2, NULL, t2, NULL);

  return 0;
}
