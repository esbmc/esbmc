#include <pthread.h>

int g;
pthread_mutex_t mutex;//=PTHREAD_MUTEX_INITIALIZER;

void *t1(void *arg)
{
  pthread_mutex_unlock(&mutex);
}

void *t2(void *arg)
{
  pthread_mutex_lock(&mutex);
  pthread_mutex_unlock(&mutex);
}

int main()
{
  pthread_t id1, id2, id3;

  pthread_mutex_init(&mutex, NULL);
  pthread_create(&id1, NULL, t1, NULL);
  pthread_create(&id2, NULL, t2, NULL);
}
