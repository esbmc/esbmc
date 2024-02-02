#include <pthread.h>
#include <assert.h>

int glob1 = 0;
pthread_mutex_t mutex2 = PTHREAD_MUTEX_INITIALIZER;

void *t_fun(void *arg)
{
  pthread_mutex_lock(&mutex2);
  glob1 = 5;
  pthread_mutex_unlock(&mutex2);
  pthread_mutex_lock(&mutex2);
  glob1 = 10;
  pthread_mutex_unlock(&mutex2);
  return NULL;
}

void *foo(void *arg)
{
  pthread_mutex_lock(&mutex2);
  assert(glob1 == 0 || glob1 == 5 || glob1 == 10);
  pthread_mutex_unlock(&mutex2);
  return NULL;
}

int main(void)
{
  pthread_t id1, id2;
  pthread_create(&id1, NULL, t_fun, NULL);
  pthread_create(&id2, NULL, foo, NULL);
  pthread_join(id1, NULL);
  pthread_join(id2, NULL);
  return 0;
}
