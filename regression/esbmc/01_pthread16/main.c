#include <pthread.h>

int g;
pthread_mutex_t mutex;

void *t1(void *arg)
{
  g = 1;
  pthread_mutex_lock(&mutex);
  g = 1;
  g = 0;
  pthread_mutex_unlock(&mutex);
  pthread_exit(NULL);
}

void *t2(void *arg)
{
  pthread_mutex_lock(&mutex);
  // this does not hold since g is set 1 
  // in thread t1 before locking the mutex
  assert(g == 0);
  pthread_mutex_unlock(&mutex);
  pthread_exit(NULL);
}

void *t3(void *arg)
{
  pthread_mutex_lock(&mutex);
  assert(g == 0);
  pthread_mutex_unlock(&mutex);
  pthread_exit(NULL);
}

int main()
{
  pthread_t id1, id2, id3;

  pthread_create(&id1, NULL, t1, NULL);
  pthread_create(&id2, NULL, t2, NULL);
  pthread_create(&id3, NULL, t3, NULL);

  return 0;
}
